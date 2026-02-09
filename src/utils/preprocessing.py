"""Image preprocessing for Car Racing environment.

Handles grayscale conversion, normalization, and frame stacking.
"""

import numpy as np
from collections import deque
import cv2


class FrameProcessor:
    """Preprocesses raw environment observations."""

    def __init__(self, frame_size=(96, 96), grayscale=True, normalize=True, frame_stack=4):
        """Initialize the frame processor.

        Args:
            frame_size (tuple): Target size (height, width)
            grayscale (bool): Convert to grayscale
            normalize (bool): Normalize pixel values to [0, 1]
            frame_stack (int): Number of frames to stack
        """
        self.frame_size = frame_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.frame_stack = frame_stack

        # Frame buffer for stacking
        self.frames = deque(maxlen=frame_stack)

    def process_frame(self, frame):
        """Process a single frame.

        Args:
            frame (array): Raw frame from environment (H, W, C)

        Returns:
            array: Processed frame
        """
        # Resize
        if frame.shape[:2] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        if self.grayscale:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Add channel dimension
            frame = np.expand_dims(frame, axis=-1)

        # Normalize
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0

        return frame

    def reset(self, initial_frame):
        """Reset the frame buffer with an initial frame.

        Args:
            initial_frame (array): First frame of episode

        Returns:
            array: Stacked frames (frame_stack, H, W, 1) or (frame_stack, H, W)
        """
        processed = self.process_frame(initial_frame)

        # Fill buffer with first frame repeated
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(processed)

        return self._get_stacked_state()

    def step(self, frame):
        """Add a new frame and return stacked state.

        Args:
            frame (array): New frame from environment

        Returns:
            array: Stacked frames
        """
        processed = self.process_frame(frame)
        self.frames.append(processed)

        return self._get_stacked_state()

    def _get_stacked_state(self):
        """Stack frames along the channel dimension.

        Returns:
            array: Stacked frames (H, W, frame_stack)
        """
        # Stack frames: (H, W, 1) * frame_stack -> (H, W, frame_stack)
        stacked = np.concatenate(list(self.frames), axis=-1)
        return stacked

    def get_state_shape(self):
        """Get the shape of processed states.

        Returns:
            tuple: State shape (H, W, frame_stack)
        """
        if self.grayscale:
            return (*self.frame_size, self.frame_stack)
        else:
            return (*self.frame_size, 3 * self.frame_stack)
