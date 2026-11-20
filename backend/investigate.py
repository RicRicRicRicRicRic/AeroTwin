import os

base = 'C:/Users/ricmi/Documents/GitHub/AeroTwin/data/processed/frames'
for run_dir in sorted(os.listdir(base)):
    if 'Facade' in run_dir:
        run_path = os.path.join(base, run_dir)
        for sub in sorted(os.listdir(run_path)):
            sub_path = os.path.join(run_path, sub)
            masks = sorted([f for f in os.listdir(sub_path) if f.startswith('mask_') and f.endswith('.png')])
            if masks:
                mask_id = masks[0].replace('mask_', '').replace('.png', '')
                frame_file = f'frame_{mask_id}.jpg'
                frame_path = os.path.join(sub_path, frame_file)
                print(f'Run: {run_dir}/{sub}')
                print(f'  First mask: {masks[0]}')
                print(f'  Frame ID: {mask_id}')
                print(f'  Frame exists: {os.path.exists(frame_path)}')
                print(f'  Full frame path: {frame_path}')
                # Check the directory structure
                parts = sub_path.replace('\\\\', '/').split('/')
                print(f'  Path parts: {parts}')
                print()