import pandas as pd
import time
import random

class SmartSensorStreamer:
    def __init__(self, csv_path, time_scale=1.0, noise_ratio=(5, 20)):
        """
        time_scale: playback speed
        noise_ratio: (min, max) number of noise samples between gestures
        """
        self.df = pd.read_csv(csv_path, comment='#')
        self.time_scale = time_scale
        self.noise_ratio = noise_ratio

        self.noise_pool = []
        self.gesture_chunks = []

        self._prepare_chunks()

    def _prepare_chunks(self):
        current_chunk = []

        for _, row in self.df.iterrows():
            if row['Button'] == 1:
                current_chunk.append(row)
            else:
                # Save any active gesture chunk
                if current_chunk:
                    self.gesture_chunks.append(pd.DataFrame(current_chunk))
                    current_chunk = []

                self.noise_pool.append(row)

        # Catch final chunk
        if current_chunk:
            self.gesture_chunks.append(pd.DataFrame(current_chunk))

        print(f"Loaded {len(self.gesture_chunks)} gesture chunks")
        print(f"Noise samples: {len(self.noise_pool)}")

    def _get_noise_block(self):
        n = random.randint(*self.noise_ratio)
        return random.choices(self.noise_pool, k=n)

    def _stream_block(self, block, callback):
        prev_time = None

        for _, row in (block.iterrows() if isinstance(block, pd.DataFrame) else enumerate(block)):
            row = row if isinstance(row, pd.Series) else row

            data = row.to_dict()

            if callback:
                callback(data)
            else:
                print(data)

            # Timing
            if prev_time is not None:
                dt = (data['time_ms'] - prev_time) / 1000.0
                if dt < 0 or dt > 1:
                    dt = 0.05  # fallback if weird jump
            else:
                dt = 0.05

            time.sleep(dt / self.time_scale)
            prev_time = data['time_ms']

    def stream(self, callback=None):
        while True:
            # Shuffle gesture order each run
            random.shuffle(self.gesture_chunks)

            for gesture in self.gesture_chunks:
                # Add random noise before gesture
                noise_block = self._get_noise_block()
                self._stream_block(noise_block, callback)

                # Stream gesture (unaltered)
                self._stream_block(gesture, callback)