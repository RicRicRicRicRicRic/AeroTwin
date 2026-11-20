import os

base = 'C:/Users/ricmi/Documents/GitHub/AeroTwin/data/processed/frames'
for run_dir in sorted(os.listdir(base)):
    if 'Facade' in run_dir:
        run_path = os.path.join(base, run_dir)
        for sub in sorted(os.listdir(run_path)):
            sub_path = os.path.join(run_path, sub)
            masks = sorted([f for f in os.listdir(sub_path) if f.startswith('mask_') and f.endswith('.png')])
            if masks:
                # Get the frame ID from the first mask
                mask_id = masks[0].replace('mask_', '').replace('.png', '')
                # Check if corresponding frame exists
                frame_file = f'frame_{mask_id}.jpg'
                frame_path = os.path.join(sub_path, frame_file)
                frame_exists = os.path.exists(frame_path)
                print(f'Run: {run_dir}/{sub}')
                print(f'  First mask: {masks[0]}')
                print(f'  Frame ID: {mask_id}')
                print(f'  Frame file: {frame_file} exists: {frame_exists}')
                if frame_exists:
                    # List a few frames
                    frames = sorted([f for f in os.listdir(sub_path) if f.startswith('frame_') and f.endswith('.jpg')])
                    print(f'  First 3 frames: {frames[:3]}')
                print()