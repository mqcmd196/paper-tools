#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import cv2
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import B5
from reportlab.lib.utils import ImageReader


def parse_time(time_str: str) -> float:
    """Parse time string and convert to seconds.
    Supports formats:
    - '100' -> 100 seconds
    - '20:10' -> 20 minutes 10 seconds = 1210 seconds
    - '20:10.5' -> 20 minutes 10.5 seconds = 1210.5 seconds
    - '1:10:05' -> 1 hour 10 minutes 5 seconds = 4205 seconds
    """
    time_str = time_str.strip()

    # If no colon, treat as seconds
    if ':' not in time_str:
        return float(time_str)

    # Split by colon
    parts = time_str.split(':')

    if len(parts) == 2:
        # mm:ss format
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # hh:mm:ss format
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def seconds_label(t_sec: float, step_sec: float) -> str:
    """Build timestamp label 't = n (s)'. Use int if step is an integer, else 2 decimals."""
    if float(step_sec).is_integer():
        return f"t = {int(round(t_sec))} (s)"
    return f"t = {t_sec:.2f} (s)"


def video_duration_sec(cap: cv2.VideoCapture) -> float:
    """Return video duration in seconds from FPS and frame count. 0.0 if unknown."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frames <= 0:
        return 0.0
    return frames / fps


def read_frame_at_time(cap: cv2.VideoCapture, t_sec: float, fps: float):
    """Seek to time t_sec (by frame index) and return a Pillow RGB Image, or None if failed."""
    if fps <= 0:
        return None
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def extract_frames_to_pdf(
    video_path: str, start_sec: float, end_sec: float, step_sec: float, cols: int = 3, rows: int = 4, custom_times: list = None
):
    """Extract frames every step_sec between start_sec and end_sec and compose PDFs in a grid layout.
    Images are arranged in a grid (cols x rows) with timestamps. Each page is saved as a separate PDF file.
    Page width is fixed to B5 width, height is calculated dynamically with minimal vertical spacing.
    Images are scaled to fit the cell width while preserving aspect ratio.
    Timestamps are displayed relative to start_sec (starting from 0).
    Additional custom time points can be specified via custom_times parameter.
    """
    # --- basic validations ---
    if step_sec <= 0:
        raise ValueError("Interval (seconds) must be > 0.")
    if start_sec < 0:
        start_sec = 0.0
    if end_sec < 0:
        raise ValueError("End time (seconds) must be >= 0.")
    if end_sec < start_sec:
        raise ValueError("End time must be >= start time.")
    if cols <= 0 or rows <= 0:
        raise ValueError("Columns and rows must be > 0.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError("Failed to get FPS from the video.")

        duration = video_duration_sec(cap)
        if duration > 0:
            end_sec = min(end_sec, duration)

        # --- build sampling times with floating-point safety ---
        times = []
        t = start_sec
        eps = 1e-9
        while t <= end_sec + eps:
            times.append(min(t, end_sec))
            t += step_sec

        # Add custom times if specified
        if custom_times:
            for custom_t in custom_times:
                if custom_t >= 0:  # only add valid times
                    times.append(custom_t)

        # Sort and remove duplicates
        times = sorted(set(times))

        if not times:
            raise RuntimeError("No frames to extract. Check range and interval.")

        # --- prepare output file path template ---
        in_path = Path(video_path)
        frames_per_page = cols * rows

        # text/spacing parameters
        label_font = "Helvetica"
        label_size = 10
        label_pad = 3  # gap between image bottom and label
        margin_h = 10  # horizontal margin
        margin_v = 5  # minimal vertical margin
        grid_gap_h = 5  # horizontal gap between grid cells
        grid_gap_v = 3  # minimal vertical gap between grid cells

        # Use B5 width, but calculate height dynamically
        b5_w, _ = B5
        page_w = b5_w

        # Calculate cell width based on B5 width
        available_w = page_w - 2 * margin_h - (cols - 1) * grid_gap_h
        cell_w = available_w / cols

        # Get first frame to determine scaling
        first_img = read_frame_at_time(cap, times[0], fps)
        if first_img is None:
            raise RuntimeError("Failed to read first frame")

        img_w, img_h = first_img.size

        # Calculate image height when scaled to fit cell width
        scale = cell_w / float(img_w)
        scaled_img_h = img_h * scale

        # label_height needs to accommodate the full font height
        label_height = label_size + label_pad * 2
        cell_h = scaled_img_h + label_height

        # Calculate page height based on rows
        page_h = 2 * margin_v + rows * cell_h + (rows - 1) * grid_gap_v

        # process frames and create PDFs
        page_num = 1
        frame_idx = 0
        c = None

        for tval in times:
            # read image
            img = read_frame_at_time(cap, tval, fps)
            if img is None:
                print(f"[warn] Failed to grab frame at t={tval:.3f}s; skipped.")
                continue

            iw, ih = img.size
            if iw <= 0 or ih <= 0:
                print(f"[warn] Invalid image size at t={tval:.3f}s; skipped.")
                continue

            # start new page if needed
            if frame_idx % frames_per_page == 0:
                # save previous page if exists
                if c is not None:
                    c.save()
                    print(f"Saved: {out_pdf}")

                # create new PDF file
                out_pdf = in_path.with_name(
                    f"{in_path.stem}_frames_{int(round(start_sec))}-{int(round(end_sec))}_every{step_sec:g}s_page{page_num}.pdf"
                )
                c = canvas.Canvas(str(out_pdf), pagesize=(page_w, page_h))
                c.setFont(label_font, label_size)
                page_num += 1

            # calculate grid position (0-indexed within current page)
            pos_in_page = frame_idx % frames_per_page
            col = pos_in_page % cols
            row = pos_in_page // cols

            # calculate cell position (top-left corner)
            x_cell = margin_h + col * (cell_w + grid_gap_h)
            y_cell = page_h - margin_v - row * (cell_h + grid_gap_v) - cell_h

            # calculate image size to fit cell (preserve aspect ratio, leave room for label)
            available_img_h = cell_h - label_height
            available_img_w = cell_w

            scale_w = available_img_w / float(iw)
            scale_h = available_img_h / float(ih)
            scale = min(scale_w, scale_h)

            draw_w = iw * scale
            draw_h = ih * scale

            # center image in cell horizontally
            x_img = x_cell + (cell_w - draw_w) / 2.0
            y_img = y_cell + label_height  # leave space for label at bottom

            # draw image
            c.drawImage(
                ImageReader(img),
                x_img,
                y_img,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                anchor="sw",
            )

            # draw timestamp label centered at bottom of cell (relative to start_sec)
            relative_time = tval - start_sec
            label = seconds_label(relative_time, step_sec)
            x_label_center = x_cell + cell_w / 2.0
            y_label = y_cell + label_pad  # position label baseline
            c.drawCentredString(x_label_center, y_label, label)

            frame_idx += 1

        # save final PDF
        if c is not None:
            c.save()
            print(f"Saved: {out_pdf}")

    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames every N seconds from a video and compose them into PDFs in a grid layout (B5 width, dynamic height)."
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("start", type=parse_time, help="Start time (seconds or hh:mm:ss format)")
    parser.add_argument("end", type=parse_time, help="End time (seconds or hh:mm:ss format)")
    parser.add_argument("interval", type=parse_time, help="Sampling interval (seconds or hh:mm:ss format)")
    parser.add_argument(
        "--cols", type=int, default=3, help="Number of columns in grid layout (default: 3)"
    )
    parser.add_argument(
        "--rows", type=int, default=4, help="Number of rows in grid layout (default: 4)"
    )
    parser.add_argument(
        "--custom-times", type=parse_time, nargs='+', help="Additional custom time points (seconds or hh:mm:ss format, space-separated)"
    )
    args = parser.parse_args()

    extract_frames_to_pdf(
        video_path=args.video,
        start_sec=args.start,
        end_sec=args.end,
        step_sec=args.interval,
        cols=args.cols,
        rows=args.rows,
        custom_times=args.custom_times,
    )


if __name__ == "__main__":
    main()
