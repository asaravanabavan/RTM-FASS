VIDEO_CONFIGURATION = {
    'maximum_height': 720,
    'target_frames_per_second': 30,
    'buffer_size': 30,
    'output_size': (1280, 720),
    'skip_frames': 2 
}

STRIKE_CONFIGURATION = {
    'sequence_length': 15,
    'velocity_threshold': 200.0,
    'cooldown_frames': 15,
    'confidence_threshold': 0.6
}

OUTPUTS_DIRECTORY = 'outputs/'
MODELS_DIRECTORY = 'models/'
DATA_DIRECTORY = 'data/'