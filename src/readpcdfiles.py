import numpy as np

def read_pcd_points(filepath):
    """Parses a PCD (Point Cloud Data) file and returns its points as a NumPy array.
    Supports both 'binary' and 'ascii' formats.
    """
    with open(filepath, 'rb') as f:
        header = []
        data_type = 'ascii'
        num_points = 0
        
        # Parse ASCII header
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header.append(line)
            if line.startswith('POINTS'):
                num_points = int(line.split()[1])
            elif line.startswith('DATA'):
                data_type = line.split()[1].lower()
                break
        
        # Read point data based on format
        if data_type == 'binary':
            # Binary payload consists of float32 values for each point
            data_bytes = f.read()
            # In standard PCD x y z fields are float32 (4 bytes each), total 12 bytes per point
            pts = np.frombuffer(data_bytes, dtype=np.float32).reshape(-1, 3)
            if len(pts) > num_points:
                pts = pts[:num_points]
            return pts
        elif data_type == 'ascii':
            # ASCII text data
            lines = f.readlines()
            pts = []
            for line in lines:
                parts = line.decode('utf-8', errors='ignore').strip().split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.array(pts, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported PCD data format: {data_type}")
