import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from anthropic import Anthropic
import numpy as np
from datetime import datetime


@dataclass
class ScraperData:
    node_id: str
    gscpi: float
    trade: float
    news: str
    political: str
    weather: str
    reporter_credibility: Dict[str, float]


@dataclass
class RiskVector:
    node_id: str
    gscpi_risk: float
    news_risk: float
    political_risk: float
    trade_risk: float
    weather_risk: float
    reporter_confidence: float
    timestamp: str


class IntelligenceProcessor:
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env variable.")
        
        self.client = Anthropic(api_key=self.api_key)
        
        self.gscpi_min = 0.0
        self.gscpi_max = 3.0
        self.trade_min = 0.0
        self.trade_max = 10_000_000
    
    def normalize_gscpi(self, gscpi_value: float) -> float:
        normalized = (gscpi_value - self.gscpi_min) / (self.gscpi_max - self.gscpi_min)
        return np.clip(normalized, 0.0, 1.0)
    
    def normalize_trade(self, trade_volume: float) -> float:
        normalized = (trade_volume - self.trade_min) / (self.trade_max - self.trade_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        return 1.0 - normalized
    
    def analyze_text_with_claude(self, nodes_data: List[ScraperData]) -> Dict[str, Dict[str, float]]:
        batch_context = self._build_batch_prompt(nodes_data)
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.3,
                system=self._get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": batch_context
                    }
                ]
            )

            response_text = message.content[0].text
            risk_scores = self._parse_claude_response(response_text)
            
            return risk_scores
            
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return {
                node.node_id: {
                    "news_risk": 0.5,
                    "political_risk": 0.5,
                    "weather_risk": 0.5
                }
                for node in nodes_data
            }
    
    def _get_system_prompt(self) -> str:
        return """You are an expert supply chain risk analyst. Your task is to analyze news, political, and weather information for various cities/ports and output risk scores.

For each location, analyze:
1. NEWS: Strikes, disruptions, port closures, infrastructure issues
2. POLITICAL: Sanctions, conflicts, regulatory changes, instability
3. WEATHER: Storms, typhoons, extreme conditions affecting logistics

Output ONLY a JSON object with this exact structure:
{
  "node_id_1": {
    "news_risk": 0.0-1.0,
    "political_risk": 0.0-1.0,
    "weather_risk": 0.0-1.0
  },
  "node_id_2": { ... }
}

Risk scale:
- 0.0-0.3: Low risk (normal operations)
- 0.3-0.6: Moderate risk (minor delays possible)
- 0.6-0.8: High risk (significant disruptions likely)
- 0.8-1.0: Severe risk (avoid if possible)

Be concise and objective. Output ONLY the JSON, no explanation."""
    
    def _build_batch_prompt(self, nodes_data: List[ScraperData]) -> str:
        prompt = "Analyze the following supply chain locations:\n\n"
        
        for node in nodes_data:
            prompt += f"{node.node_id}\n"
            prompt += f"NEWS: {node.news[:500]}...\n"
            prompt += f"POLITICAL: {node.political[:500]}...\n"
            prompt += f"WEATHER: {node.weather[:300]}...\n\n"
        
        prompt += "\nProvide risk scores for all locations in JSON format."
        return prompt
    
    def _parse_claude_response(self, response: str) -> Dict[str, Dict[str, float]]:
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            risk_scores = json.loads(response.strip())
            return risk_scores
            
        except json.JSONDecodeError as e:
            print(f"Error parsing Claude response: {e}")
            print(f"Response: {response}")
            return {}
    
    def apply_reporter_weights(self, 
                               base_risk: float, 
                               credibility: float) -> float:
        adjusted_risk = base_risk * credibility + 0.5 * (1 - credibility)
        return np.clip(adjusted_risk, 0.0, 1.0)
    
    def process_batch(self, nodes_data: List[ScraperData]) -> List[RiskVector]:
        print(f"Processing {len(nodes_data)} nodes")
        
        print("Step 1: Normalizing GSCPI and Trade data")
        gscpi_risks = {
            node.node_id: self.normalize_gscpi(node.gscpi)
            for node in nodes_data
        }
        trade_risks = {
            node.node_id: self.normalize_trade(node.trade)
            for node in nodes_data
        }
        print("Step 2: Analyzing text data with Claude API")
        claude_risks = self.analyze_text_with_claude(nodes_data)
        
        print("Step 3: Applying Reporter credibility weights")
        risk_vectors = []
        timestamp = datetime.utcnow().isoformat()
        
        for node in nodes_data:
            gscpi_risk = gscpi_risks[node.node_id]
            trade_risk = trade_risks[node.node_id]
            
            claude_risk = claude_risks.get(node.node_id, {
                "news_risk": 0.5,
                "political_risk": 0.5,
                "weather_risk": 0.5
            })
            reporter_confidence = np.mean(list(node.reporter_credibility.values()))
            
            news_risk = self.apply_reporter_weights(
                claude_risk["news_risk"],
                node.reporter_credibility.get("news", 0.5)
            )
            political_risk = self.apply_reporter_weights(
                claude_risk["political_risk"],
                node.reporter_credibility.get("political", 0.5)
            )
            weather_risk = self.apply_reporter_weights(
                claude_risk["weather_risk"],
                node.reporter_credibility.get("weather", 0.5)
            )
            
            risk_vector = RiskVector(
                node_id=node.node_id,
                gscpi_risk=gscpi_risk,
                news_risk=news_risk,
                political_risk=political_risk,
                trade_risk=trade_risk,
                weather_risk=weather_risk,
                reporter_confidence=reporter_confidence,
                timestamp=timestamp
            )
            
            risk_vectors.append(risk_vector)
        
        print(f"Processed {len(risk_vectors)} risk vectors")
        return risk_vectors
    
    def export_to_json(self, risk_vectors: List[RiskVector], filepath: str):
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_nodes": len(risk_vectors),
            "risk_vectors": [
                {
                    "node_id": rv.node_id,
                    "gscpi_risk": float(rv.gscpi_risk),
                    "news_risk": float(rv.news_risk),
                    "political_risk": float(rv.political_risk),
                    "trade_risk": float(rv.trade_risk),
                    "weather_risk": float(rv.weather_risk),
                    "reporter_confidence": float(rv.reporter_confidence)
                }
                for rv in risk_vectors
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported risk vectors to {filepath}")

if __name__ == "__main__":
    sample_data = [
        ScraperData(
            node_id="Hong_Kong",
            gscpi=1.45,
            trade=8500000,
            news="Port workers in Hong Kong threaten strike over wages. Container terminals may face delays.",
            political="Political tensions remain high. New regulations on shipping compliance introduced.",
            weather="Typhoon Saola approaching. Category 4 storm expected to make landfall in 48 hours.",
            reporter_credibility={"news": 0.85, "political": 0.75, "weather": 0.95}
        ),
        ScraperData(
            node_id="Singapore",
            gscpi=0.85,
            trade=12000000,
            news="Singapore port reports record throughput. Operations running smoothly.",
            political="Stable political environment. Trade agreements with EU renewed.",
            weather="Clear conditions. No weather disruptions expected.",
            reporter_credibility={"news": 0.90, "political": 0.85, "weather": 0.90}
        ),
        ScraperData(
            node_id="Shanghai",
            gscpi=1.20,
            trade=9500000,
            news="COVID restrictions lifted at Shanghai port. Gradual return to normal operations.",
            political="Government announces new export controls on technology goods.",
            weather="Heavy fog affecting vessel movements. Delays of 4-6 hours reported.",
            reporter_credibility={"news": 0.80, "political": 0.70, "weather": 0.85}
        )
    ]
    print("Supply Chain Intelligence Processor\n")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Using mock mode.")
        print("To run with real Claude API, set: export ANTHROPIC_API_KEY='your-key'\n")

        processor = IntelligenceProcessor(api_key="mock-key")
        processor.client = None
        risk_vectors = []
        for node in sample_data:
            risk_vectors.append(RiskVector(
                node_id=node.node_id,
                gscpi_risk=processor.normalize_gscpi(node.gscpi),
                news_risk=0.7 if "strike" in node.news.lower() else 0.3,
                political_risk=0.6 if "tensions" in node.political.lower() else 0.3,
                trade_risk=processor.normalize_trade(node.trade),
                weather_risk=0.85 if "typhoon" in node.weather.lower() else 0.2,
                reporter_confidence=0.85,
                timestamp=datetime.utcnow().isoformat()
            ))
    else:
        processor = IntelligenceProcessor()
        risk_vectors = processor.process_batch(sample_data)
    
    print("\nRisk Vectors Generated\n")
    for rv in risk_vectors:
        print(f"{rv.node_id}:")
        print(f"  GSCPI Risk:     {rv.gscpi_risk:.3f}")
        print(f"  News Risk:      {rv.news_risk:.3f}")
        print(f"  Political Risk: {rv.political_risk:.3f}")
        print(f"  Trade Risk:     {rv.trade_risk:.3f}")
        print(f"  Weather Risk:   {rv.weather_risk:.3f}")
        print(f"  Reporter Conf:  {rv.reporter_confidence:.3f}")
        overall = np.mean([rv.gscpi_risk, rv.news_risk, rv.political_risk, 
                          rv.trade_risk, rv.weather_risk])
        print(f"  Overall Risk:   {overall:.3f}")
        print()
    processor.export_to_json(risk_vectors, "risk_vectors_output.json")
    print("\nIntelligence processing complete")