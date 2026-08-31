The scene requires undistortion using the provided camera parameters, undistortion scripts, and frame-0 calibration. Use the scripts below to copy parameters to other frames and run COLMAP undistortion.

Run all commands from this directory:

```bash
cd utils/multiview_data_process
```

## Steps

1. Convert multi-view videos (`.mp4`) into per-frame images (`.png`). Edit the number of viewpoints (line 32) and video path (line 34) in `video2stream.py`:

   ```bash
   python video2stream.py
   ```

2. Undistort and calibrate **frame 0**. If you use our provided cameras, skip this step after placing the camera files and distorted images in `images/` under `/path/dataset/VRU_long/frame000000`:

   ```bash
   python convert.py -s /path/dataset/VRU_long/frame000000
   ```

3. Copy camera parameters from `frame000000` to all other frames, and copy undistortion parameters to the parent scene directory:

   ```bash
   python copy_cams.py --source /path/dataset/VRU_long/frame000000 --scene /path/dataset/VRU_long
   ```

4. Undistort the remaining frames and write results to each frame's `images/` folder. For sequences longer than 250 frames, also set `--last_frame_id` in `convert_frames.py`:

   ```bash
   python convert_frames.py -s /path/dataset/VRU_long
   ```
