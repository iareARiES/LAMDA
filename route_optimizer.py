import json
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from math import radians, cos, sin, asin, sqrt


@dataclass
class Route:
    path: List[str]
    total_distance_km: float
    total_risk: float
    avg_risk: float
    estimated_time_hours: float
    route_score: float


class RouteOptimizer:
    def __init__(self, graph_state_file: str):
        with open(graph_state_file, 'r') as f:
            self.graph_state = json.load(f)
        self.nodes = {
            node["node_id"]: {
                "latitude": node["latitude"],
                "longitude": node["longitude"],
                "overall_risk": node["overall_risk"]
            }
            for node in self.graph_state["nodes"]
        }
        self.adjacency = {node_id: [] for node_id in self.nodes.keys()}
        for edge in self.graph_state["edges"]:
            self.adjacency[edge["source"]].append({
                "target": edge["target"],
                "distance_km": edge["distance_km"],
                "trade_volume": edge.get("trade_volume", 1000000)
            })
            self.adjacency[edge["target"]].append({
                "target": edge["source"],
                "distance_km": edge["distance_km"],
                "trade_volume": edge.get("trade_volume", 1000000)
            })
        
        print(f"Route Optimizer initialized with {len(self.nodes)} nodes")
    
    def haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        r = 6371
        
        return c * r
    
    def calculate_heuristic(self, current: str, goal: str) -> float:
        curr_node = self.nodes[current]
        goal_node = self.nodes[goal]
        
        distance = self.haversine_distance(
            curr_node["latitude"], curr_node["longitude"],
            goal_node["latitude"], goal_node["longitude"]
        )
        
        avg_risk = (curr_node["overall_risk"] + goal_node["overall_risk"]) / 2
        
        return distance * (1 + avg_risk)
    
    def calculate_edge_cost(self, 
                           source: str, 
                           target: str, 
                           edge_info: Dict,
                           weights: Dict[str, float]) -> float:
        distance = edge_info["distance_km"]
        trade_volume = edge_info["trade_volume"]
        
        source_risk = self.nodes[source]["overall_risk"]
        target_risk = self.nodes[target]["overall_risk"]
        avg_risk = (source_risk + target_risk) / 2
        
        distance_cost = distance * weights["distance"]
        risk_cost = avg_risk * distance * weights["risk"]
        
        trade_factor = 1.0 / (1.0 + trade_volume / 1_000_000)
        trade_cost = distance * trade_factor * weights["trade"]
        
        total_cost = distance_cost + risk_cost + trade_cost
        
        return total_cost
    
    def find_optimal_route(self,
                          source: str,
                          destination: str,
                          weights: Optional[Dict[str, float]] = None) -> Optional[Route]:
        if weights is None:
            weights = {
                "risk": 1.0,
                "distance": 0.3,
                "trade": 0.2
            }
        
        if source not in self.nodes or destination not in self.nodes:
            print(f"Invalid source or destination")
            return None
        
        open_set = []
        heapq.heappush(open_set, (0, source))
        
        came_from = {}
        g_score = {node_id: float('inf') for node_id in self.nodes}
        g_score[source] = 0
        
        f_score = {node_id: float('inf') for node_id in self.nodes}
        f_score[source] = self.calculate_heuristic(source, destination)
        
        distance_to = {source: 0.0}
        risk_to = {source: 0.0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == destination:
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(source)
                path.reverse()
                
                total_distance = distance_to[destination]
                total_risk = risk_to[destination]
                avg_risk = total_risk / len(path)
                
                estimated_time = total_distance / 37.0
                
                route_score = g_score[destination]
                
                route = Route(
                    path=path,
                    total_distance_km=total_distance,
                    total_risk=total_risk,
                    avg_risk=avg_risk,
                    estimated_time_hours=estimated_time,
                    route_score=route_score
                )
                
                return route
            
            for neighbor_info in self.adjacency[current]:
                neighbor = neighbor_info["target"]
                
                edge_cost = self.calculate_edge_cost(
                    current, neighbor, neighbor_info, weights
                )
                tentative_g_score = g_score[current] + edge_cost
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.calculate_heuristic(
                        neighbor, destination
                    )
                    
                    distance_to[neighbor] = distance_to[current] + neighbor_info["distance_km"]
                    node_risk = self.nodes[neighbor]["overall_risk"]
                    risk_to[neighbor] = risk_to[current] + node_risk
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def find_k_best_routes(self,
                          source: str,
                          destination: str,
                          k: int = 3,
                          diversity_penalty: float = 0.3) -> List[Route]:
        routes = []
        used_nodes_count = {node_id: 0 for node_id in self.nodes}
        
        weight_configs = [
            {"risk": 1.0, "distance": 0.3, "trade": 0.2},
            {"risk": 0.5, "distance": 0.8, "trade": 0.2},
            {"risk": 0.7, "distance": 0.4, "trade": 0.5},
            {"risk": 1.2, "distance": 0.2, "trade": 0.1},
            {"risk": 0.3, "distance": 1.0, "trade": 0.3},
        ]
        
        for i in range(min(k, len(weight_configs))):
            route = self.find_optimal_route(source, destination, weight_configs[i])
            
            if route is None:
                break
            
            is_diverse = True
            for existing_route in routes:
                overlap = len(set(route.path) & set(existing_route.path))
                if overlap / len(route.path) > 0.7:
                    is_diverse = False
                    break
            
            if is_diverse:
                routes.append(route)
                
                for node in route.path:
                    used_nodes_count[node] += 1
        
        routes.sort(key=lambda r: r.route_score)
        
        return routes[:k]
    
    def analyze_route(self, route: Route) -> Dict:
        analysis = {
            "route_summary": {
                "path": " → ".join(route.path),
                "total_distance_km": round(route.total_distance_km, 2),
                "avg_risk": round(route.avg_risk, 3),
                "estimated_time_days": round(route.estimated_time_hours / 24, 1),
                "route_score": round(route.route_score, 2)
            },
            "node_risks": [],
            "high_risk_segments": [],
            "bottlenecks": []
        }
        
        for node_id in route.path:
            node = self.nodes[node_id]
            analysis["node_risks"].append({
                "node": node_id,
                "risk": round(node["overall_risk"], 3)
            })
            
            if node["overall_risk"] > 0.7:
                analysis["high_risk_segments"].append({
                    "node": node_id,
                    "risk": round(node["overall_risk"], 3),
                    "warning": "High risk - consider alternative"
                })
        
        for i, node_id in enumerate(route.path[1:-1], 1):
            num_alternatives = len([
                n for n in self.adjacency[node_id]
                if n["target"] in route.path[i-1:i+2]
            ])
            
            if num_alternatives <= 2:
                analysis["bottlenecks"].append({
                    "node": node_id,
                    "alternatives": num_alternatives,
                    "warning": "Limited alternative routes"
                })
        
        return analysis

if __name__ == "__main__":
    print("Route Optimizer\n")
    
    import os
    if not os.path.exists("graph_state.json"):
        print("graph_state.json not found. Run graph_risk_engine.py first.")
        print("Creating sample graph for demo\n")
        
        sample_state = {
            "timestamp": "2024-01-01T00:00:00",
            "nodes": [
                {"node_id": "Hong_Kong", "latitude": 22.3193, "longitude": 114.1694, 
                 "overall_risk": 0.65, "risk_vector": [0.6, 0.7, 0.6, 0.5, 0.8, 0.85]},
                {"node_id": "Singapore", "latitude": 1.3521, "longitude": 103.8198, 
                 "overall_risk": 0.25, "risk_vector": [0.2, 0.3, 0.2, 0.3, 0.2, 0.90]},
                {"node_id": "Shanghai", "latitude": 31.2304, "longitude": 121.4737, 
                 "overall_risk": 0.45, "risk_vector": [0.4, 0.5, 0.4, 0.5, 0.4, 0.80]},
                {"node_id": "Tokyo", "latitude": 35.6762, "longitude": 139.6503, 
                 "overall_risk": 0.30, "risk_vector": [0.3, 0.3, 0.3, 0.3, 0.3, 0.85]},
                {"node_id": "Los_Angeles", "latitude": 34.0522, "longitude": -118.2437, 
                 "overall_risk": 0.35, "risk_vector": [0.3, 0.4, 0.3, 0.4, 0.3, 0.90]}
            ],
            "edges": [
                {"source": "Hong_Kong", "target": "Singapore", "distance_km": 2590, "trade_volume": 5000000},
                {"source": "Hong_Kong", "target": "Shanghai", "distance_km": 1213, "trade_volume": 8000000},
                {"source": "Singapore", "target": "Shanghai", "distance_km": 3898, "trade_volume": 4000000},
                {"source": "Shanghai", "target": "Tokyo", "distance_km": 1768, "trade_volume": 6000000},
                {"source": "Tokyo", "target": "Los_Angeles", "distance_km": 8806, "trade_volume": 7000000},
                {"source": "Singapore", "target": "Los_Angeles", "distance_km": 14100, "trade_volume": 3000000}
            ]
        }
        
        with open("graph_state.json", "w") as f:
            json.dump(sample_state, f, indent=2)
    
    optimizer = RouteOptimizer("graph_state.json")
    
    print("Finding optimal routes: Hong Kong → Los Angeles")
    
    routes = optimizer.find_k_best_routes("Hong_Kong", "Los_Angeles", k=3)
    
    if not routes:
        print("No routes found between these locations")
    else:
        for i, route in enumerate(routes, 1):
            print(f"Route {i}:")
            print(f"  Path: {' → '.join(route.path)}")
            print(f"  Distance: {route.total_distance_km:.0f} km")
            print(f"  Average Risk: {route.avg_risk:.3f}")
            print(f"  Estimated Time: {route.estimated_time_hours/24:.1f} days")
            print(f"  Route Score: {route.route_score:.2f}")
            print()
        
        print("Detailed Analysis of Best Route")
        
        best_route = routes[0]
        analysis = optimizer.analyze_route(best_route)
        
        print(json.dumps(analysis, indent=2))
        
        routes_export = {
            "source": "Hong_Kong",
            "destination": "Los_Angeles",
            "num_routes": len(routes),
            "routes": [
                {
                    "rank": i + 1,
                    "path": route.path,
                    "total_distance_km": route.total_distance_km,
                    "avg_risk": route.avg_risk,
                    "estimated_time_hours": route.estimated_time_hours,
                    "route_score": route.route_score
                }
                for i, route in enumerate(routes)
            ]
        }
        
        with open("optimized_routes.json", "w") as f:
            json.dump(routes_export, f, indent=2)
        
        print("\nRoutes exported to optimized_routes.json")
    
    print("\nRoute optimization complete!")