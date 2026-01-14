import sys
script_name = sys.argv[0]
if script_name == "-m" and len(sys.argv) > 1:
    script_name = sys.argv[1]

if not script_name.endswith('mnist_world.py'):
    from .mnist_world import MNISTWorldVideoDataset
    
if not script_name.endswith('blockworld.py'):
    from .blockworld import BlockWorldVideoDataset
