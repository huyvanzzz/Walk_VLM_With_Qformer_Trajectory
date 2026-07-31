from dataclasses import dataclass
from typing import Dict

@dataclass
class GroundTruthData:
    """Ground-truth answer text used for public training."""
    location: str
    weather: str
    traffic: str
    scene: str
    instruction: str

    def to_direct_text(self) -> str:
        """Return the final spoken guidance text."""
        return (self.instruction or "").strip()


def get_response_format(config: Dict = None) -> str:
    return "direct_text"


def format_ground_truth(metadata: Dict, response_format: str = "direct_text") -> str:
    ground_truth = map_metadata_to_ground_truth(metadata)
    return ground_truth.to_direct_text()


def map_metadata_to_ground_truth(metadata: Dict) -> GroundTruthData:
    """Map WAD metadata to the public direct-text target."""
    area_map = {
        'Pedestrian Path': 'pedestrian_path',
        'Road': 'road',
        'Corridor': 'corridor',
        'Busy Street': 'busy_street',
        'Shopping Mall': 'shopping_mall',
        'Bicycle Lane': 'bicycle_lane',
        'Restaurant': 'restaurant',
        'Other': 'other'
    }
    
    # Weather mapping
    weather_map = {
        'Sunny': 'sunny',
        'Overcast': 'overcast',
        'Cloudy': 'cloudy',
        'Night': 'night',
        'Indoor': 'indoor',
        'Other': 'other'
    }

    traffic_map = {
        'High': 'high',
        'Mid': 'moderate',
        'Low': 'low'
    }

    location = area_map.get(metadata.get('area_type', 'Other'), 'other')
    weather = weather_map.get(metadata.get('weather_condition', 'Other'), 'other')
    traffic = traffic_map.get(metadata.get('traffic_flow_rating', 'Low'), 'low')
    scene = metadata.get('summary', '')

    if metadata.get('QA') and isinstance(metadata['QA'], dict):
        instruction = metadata['QA'].get('A', '')
    elif metadata.get('alter'):
        instruction = metadata['alter']
    else:
        instruction = ''
    
    return GroundTruthData(
        location=location,
        weather=weather,
        traffic=traffic,
        scene=scene,
        instruction=instruction
    )
