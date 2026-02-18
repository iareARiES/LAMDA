import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class GraphNode:
    node_id: str
    latitude: float
    longitude: float
    risk_vector: np.ndarray


@dataclass
class GraphEdge:
    source: str
    target: str
    distance_km: float
    trade_volume: float


class GraphAttentionRiskModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 6,
                 hidden_dim: int = 16,
                 output_dim: int = 6,
                 num_heads: int = 4):

        super(GraphAttentionRiskModel, self).__init__()
        
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=0.2,
            concat=True
        )
        
        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=output_dim,
            heads=1,
            dropout=0.2,
            concat=False
        )
        
        self.temporal_decay = nn.Parameter(torch.tensor(0.9))
    
    def forward(self, x, edge_index, edge_attr=None):

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index)
        
        x = torch.sigmoid(x)
        
        return x


class GraphRiskEngine:
   
    def __init__(self, db_path: str = "supply_chain_graph.db"):

        self.db_path = db_path
        self.model = GraphAttentionRiskModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        
        self.history_length = 10
        
        self._init_database()
        
        print("Graph Risk Engine initialized")
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                latitude REAL,
                longitude REAL,
                current_risk REAL,
                last_updated TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                distance_km REAL,
                trade_volume REAL,
                UNIQUE(source, target)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT,
                timestamp TEXT,
                gscpi_risk REAL,
                news_risk REAL,
                political_risk REAL,
                trade_risk REAL,
                weather_risk REAL,
                reporter_confidence REAL,
                overall_risk REAL,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_node_time 
            ON risk_history(node_id, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        
        print("Database initialized")
    
    def load_graph_structure(self, nodes_file: str, edges_file: str):
        with open(nodes_file, 'r') as f:
            nodes_data = json.load(f)
        
        for node in nodes_data["nodes"]:
            self.nodes[node["node_id"]] = GraphNode(
                node_id=node["node_id"],
                latitude=node["latitude"],
                longitude=node["longitude"],
                risk_vector=np.zeros(6)
            )
        with open(edges_file, 'r') as f:
            edges_data = json.load(f)
        
        for edge in edges_data["edges"]:
            self.edges.append(GraphEdge(
                source=edge["source"],
                target=edge["target"],
                distance_km=edge["distance_km"],
                trade_volume=edge.get("trade_volume", 1000000)
            ))
        
        print(f"Loaded graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def update_risk_vectors(self, risk_vectors_file: str):
        with open(risk_vectors_file, 'r') as f:
            data = json.load(f)
        
        for rv in data["risk_vectors"]:
            node_id = rv["node_id"]
            if node_id in self.nodes:
                self.nodes[node_id].risk_vector = np.array([
                    rv["gscpi_risk"],
                    rv["news_risk"],
                    rv["political_risk"],
                    rv["trade_risk"],
                    rv["weather_risk"],
                    rv["reporter_confidence"]
                ])
        
        print(f"Updated {len(data['risk_vectors'])} node risk vectors")
    
    def _build_pytorch_geometric_data(self) -> Data:
        node_ids = list(self.nodes.keys())
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        
        x = torch.tensor(
            [self.nodes[nid].risk_vector for nid in node_ids],
            dtype=torch.float32
        )
        edge_index = []
        edge_weights = []
        
        for edge in self.edges:
            if edge.source in node_id_to_idx and edge.target in node_id_to_idx:
                src_idx = node_id_to_idx[edge.source]
                tgt_idx = node_id_to_idx[edge.target]
                
                edge_index.append([src_idx, tgt_idx])
                edge_index.append([tgt_idx, src_idx])
                
                weight = 1.0 / (1.0 + edge.distance_km / 1000.0)
                edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)
        
        return data, node_ids
    
    def propagate_risks(self) -> Dict[str, np.ndarray]:
        print("Running risk propagation through graph")
        
        data, node_ids = self._build_pytorch_geometric_data()
        
        self.model.eval()
        with torch.no_grad():
            updated_features = self.model(data.x, data.edge_index)
        
        updated_risks = {}
        for idx, node_id in enumerate(node_ids):
            updated_risks[node_id] = updated_features[idx].numpy()
            
            self.nodes[node_id].risk_vector = updated_features[idx].numpy()
        
        print(f"Risk propagation complete for {len(updated_risks)} nodes")
        return updated_risks
    
    def store_snapshot(self, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for node_id, node in self.nodes.items():
            rv = node.risk_vector
            overall_risk = float(np.mean(rv[:5]))
            
            cursor.execute("""
                INSERT OR REPLACE INTO nodes (node_id, latitude, longitude, current_risk, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (node_id, node.latitude, node.longitude, overall_risk, timestamp))
            
            cursor.execute("""
                INSERT INTO risk_history 
                (node_id, timestamp, gscpi_risk, news_risk, political_risk, 
                 trade_risk, weather_risk, reporter_confidence, overall_risk)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (node_id, timestamp, float(rv[0]), float(rv[1]), float(rv[2]),
                  float(rv[3]), float(rv[4]), float(rv[5]), overall_risk))
        
        cursor.execute("""
            DELETE FROM risk_history
            WHERE id NOT IN (
                SELECT id FROM risk_history
                WHERE node_id = risk_history.node_id
                ORDER BY timestamp DESC
                LIMIT ?
            )
        """, (self.history_length,))
        
        conn.commit()
        conn.close()
        
        print(f"Stored snapshot at {timestamp}")
    
    def get_node_risk_score(self, node_id: str) -> float:
        if node_id not in self.nodes:
            return 0.5
        
        rv = self.nodes[node_id].risk_vector
        return float(np.mean(rv[:5]))
    
    def export_graph_state(self, output_file: str):
        graph_state = {
            "timestamp": datetime.utcnow().isoformat(),
            "nodes": [
                {
                    "node_id": node_id,
                    "latitude": node.latitude,
                    "longitude": node.longitude,
                    "risk_vector": node.risk_vector.tolist(),
                    "overall_risk": self.get_node_risk_score(node_id)
                }
                for node_id, node in self.nodes.items()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "distance_km": edge.distance_km,
                    "trade_volume": edge.trade_volume
                }
                for edge in self.edges
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(graph_state, f, indent=2)
        
        print(f"Exported graph state to {output_file}")
    
    def get_historical_trends(self, node_id: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, gscpi_risk, news_risk, political_risk, 
                   trade_risk, weather_risk, overall_risk
            FROM risk_history
            WHERE node_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (node_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "gscpi_risk": row[1],
                "news_risk": row[2],
                "political_risk": row[3],
                "trade_risk": row[4],
                "weather_risk": row[5],
                "overall_risk": row[6]
            })
        
        return history

if __name__ == "__main__":
    print("Graph Risk Engine\n")
    
    engine = GraphRiskEngine()
    
    print("Creating sample graph structure")
    
    sample_nodes = {
        "nodes": [
            {"node_id": "Hong_Kong", "latitude": 22.3193, "longitude": 114.1694},
            {"node_id": "Singapore", "latitude": 1.3521, "longitude": 103.8198},
            {"node_id": "Shanghai", "latitude": 31.2304, "longitude": 121.4737},
            {"node_id": "Tokyo", "latitude": 35.6762, "longitude": 139.6503},
            {"node_id": "Los_Angeles", "latitude": 34.0522, "longitude": -118.2437}
        ]
    }
    
    sample_edges = {
        "edges": [
            {"source": "Hong_Kong", "target": "Singapore", "distance_km": 2590, "trade_volume": 5000000},
            {"source": "Hong_Kong", "target": "Shanghai", "distance_km": 1213, "trade_volume": 8000000},
            {"source": "Singapore", "target": "Shanghai", "distance_km": 3898, "trade_volume": 4000000},
            {"source": "Shanghai", "target": "Tokyo", "distance_km": 1768, "trade_volume": 6000000},
            {"source": "Tokyo", "target": "Los_Angeles", "distance_km": 8806, "trade_volume": 7000000}
        ]
    }
    
    with open("graph_nodes.json", "w") as f:
        json.dump(sample_nodes, f, indent=2)
    
    with open("graph_edges.json", "w") as f:
        json.dump(sample_edges, f, indent=2)
    
    engine.load_graph_structure("graph_nodes.json", "graph_edges.json")
    
    import os
    if os.path.exists("risk_vectors_output.json"):
        print("\nLoading risk vectors from Intelligence Processor")
        engine.update_risk_vectors("risk_vectors_output.json")
    else:
        print("\nrisk_vectors_output.json not found. Using random risk vectors for demo.")
        for node_id in engine.nodes.keys():
            engine.nodes[node_id].risk_vector = np.random.uniform(0.2, 0.8, 6)

    updated_risks = engine.propagate_risks()
    
    print("\nUpdated Risk Scores\n")
    for node_id, risk_vector in updated_risks.items():
        overall = np.mean(risk_vector[:5])
        print(f"{node_id}:")
        print(f"  GSCPI: {risk_vector[0]:.3f}, News: {risk_vector[1]:.3f}, "
              f"Political: {risk_vector[2]:.3f}")
        print(f"  Trade: {risk_vector[3]:.3f}, Weather: {risk_vector[4]:.3f}, "
              f"Reporter: {risk_vector[5]:.3f}")
        print(f"  Overall Risk: {overall:.3f}")
        print()
    
    engine.store_snapshot()
    
    engine.export_graph_state("graph_state.json")
    
    print("\nGraph Risk Engine processing complete")