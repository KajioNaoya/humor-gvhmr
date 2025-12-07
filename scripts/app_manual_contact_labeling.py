import os
from typing import Optional

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError as e:  # pragma: no cover - environment dependent
    raise ImportError(
        "tkinter is required for the manual contact labeling app. "
        "Please install/enable tkinter in your Python environment."
    ) from e

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError(
        "Pillow (PIL) is required for the manual contact labeling app. "
        "Install it via 'pip install pillow'."
    )


class ContactLabelApp:
    """
    Simple GUI tool for manually labeling left/right foot contacts for each frame
    of a video.

    - Load a video file.
    - Step through frames one by one.
    - For each frame, mark whether the left/right foot is in contact (1) or not (0).
    - Export the labels as a T x 2 CSV file:
        column 0: left_contact (0 or 1)
        column 1: right_contact (0 or 1)

    The CSV is written using numpy.savetxt with a commented header line:
        # left_contact,right_contact
    so it can be loaded later with numpy.loadtxt (used in demo_mmpose_external_smpl.py).
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Manual Foot Contact Labeling")

        # OpenCV video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.total_frames: int = 0
        self.current_frame_idx: int = 0

        # Contact labels: shape (T, 2) with int values 0/1
        self.contact_labels: Optional[np.ndarray] = None
        # Whether each frame has been explicitly labeled at least once
        self.labeled_mask: Optional[np.ndarray] = None

        # Tkinter variables for GUI state
        self.left_contact_var = tk.IntVar(value=0)
        self.right_contact_var = tk.IntVar(value=0)

        # UI components
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        load_btn = tk.Button(top_frame, text="Load Video", command=self.load_video)
        load_btn.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(top_frame, text="Save CSV", command=self.save_csv)
        save_btn.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(
            top_frame,
            text="No video loaded",
            anchor="w",
        )
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Canvas for video frame display
        self.canvas = tk.Canvas(self.root, width=640, height=360, bg="black")
        self.canvas.pack(side=tk.TOP, padx=5, pady=5)
        self._canvas_image = None  # keep reference to PhotoImage

        # Controls frame
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        prev_btn = tk.Button(ctrl_frame, text="<< Prev", command=self.prev_frame)
        prev_btn.pack(side=tk.LEFT, padx=5)

        next_btn = tk.Button(ctrl_frame, text="Next >>", command=self.next_frame)
        next_btn.pack(side=tk.LEFT, padx=5)

        left_check = tk.Checkbutton(
            ctrl_frame,
            text="Left foot contact (1/0)",
            variable=self.left_contact_var,
            onvalue=1,
            offvalue=0,
        )
        left_check.pack(side=tk.LEFT, padx=10)

        right_check = tk.Checkbutton(
            ctrl_frame,
            text="Right foot contact (1/0)",
            variable=self.right_contact_var,
            onvalue=1,
            offvalue=0,
        )
        right_check.pack(side=tk.LEFT, padx=10)

        # Keyboard shortcuts for faster labeling
        self.root.bind("<Left>", lambda event: self.prev_frame())
        self.root.bind("<Right>", lambda event: self.next_frame())
        self.root.bind("a", lambda event: self.toggle_left())
        self.root.bind("l", lambda event: self.toggle_right())

        help_text = (
            "Controls:\n"
            "- Load Video: select an input video file.\n"
            "- Next/Prev or Right/Left arrow keys: move between frames.\n"
            "- Checkboxes or 'a'/'l' keys: toggle left/right contact for the current frame.\n"
            "- Save CSV: export T x 2 contact labels (left, right) as CSV."
        )
        help_label = tk.Label(self.root, text=help_text, justify=tk.LEFT, anchor="w")
        help_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Video loading and frame navigation
    # ------------------------------------------------------------------
    def load_video(self) -> None:
        video_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not video_path:
            return

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Failed to open video: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            messagebox.showerror("Error", "Video appears to have no frames.")
            return

        self.cap = cap
        self.video_path = video_path
        self.total_frames = total_frames
        self.current_frame_idx = 0

        # Initialize contact labels (T x 2) with zeros
        self.contact_labels = np.zeros((self.total_frames, 2), dtype=np.int32)
        # Track which frames have been touched by the user
        self.labeled_mask = np.zeros((self.total_frames,), dtype=bool)
        self.left_contact_var.set(0)
        self.right_contact_var.set(0)

        self._update_info_label()
        self._show_current_frame()

    def _update_info_label(self) -> None:
        if self.video_path is None:
            self.info_label.config(text="No video loaded")
        else:
            fname = os.path.basename(self.video_path)
            self.info_label.config(
                text=f"{fname} | Frame {self.current_frame_idx + 1}/{self.total_frames}"
            )

    def _show_current_frame(self) -> None:
        if self.cap is None or self.total_frames == 0:
            return

        # Seek to the desired frame index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            messagebox.showerror(
                "Error", f"Failed to read frame {self.current_frame_idx}."
            )
            return

        # Convert BGR (OpenCV) -> RGB (PIL/Tkinter)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit canvas while keeping aspect ratio
        canvas_w = int(self.canvas["width"])
        canvas_h = int(self.canvas["height"])
        h, w, _ = frame_rgb.shape
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=img)
        self._canvas_image = photo  # keep reference

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=photo,
            anchor=tk.CENTER,
        )

        # Restore label state for this frame if available
        if self.contact_labels is not None:
            self.left_contact_var.set(int(self.contact_labels[self.current_frame_idx, 0]))
            self.right_contact_var.set(int(self.contact_labels[self.current_frame_idx, 1]))

        self._update_info_label()

    def _store_current_labels(self) -> None:
        if self.contact_labels is None:
            return
        if 0 <= self.current_frame_idx < self.total_frames:
            self.contact_labels[self.current_frame_idx, 0] = int(self.left_contact_var.get())
            self.contact_labels[self.current_frame_idx, 1] = int(self.right_contact_var.get())
            if self.labeled_mask is not None:
                self.labeled_mask[self.current_frame_idx] = True

    def _init_labels_from_previous_if_unlabeled(self, idx: int) -> None:
        """
        If frame `idx` has not been labeled yet, initialize its labels from
        the previous frame. This makes annotation easier when contact state
        does not change frequently.
        """
        if (
            self.contact_labels is None
            or self.labeled_mask is None
            or self.total_frames == 0
        ):
            return
        if not (0 <= idx < self.total_frames):
            return
        if self.labeled_mask[idx]:
            return
        prev_idx = idx - 1
        if 0 <= prev_idx < self.total_frames and self.labeled_mask[prev_idx]:
            self.contact_labels[idx] = self.contact_labels[prev_idx]

    def next_frame(self) -> None:
        if self.cap is None:
            return
        self._store_current_labels()
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            # If the new frame is still unlabeled, start from the previous frame's state
            self._init_labels_from_previous_if_unlabeled(self.current_frame_idx)
            self._show_current_frame()

    def prev_frame(self) -> None:
        if self.cap is None:
            return
        self._store_current_labels()
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._show_current_frame()

    def toggle_left(self) -> None:
        self.left_contact_var.set(1 - int(self.left_contact_var.get()))
        self._store_current_labels()

    def toggle_right(self) -> None:
        self.right_contact_var.set(1 - int(self.right_contact_var.get()))
        self._store_current_labels()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def save_csv(self) -> None:
        if self.contact_labels is None or self.total_frames == 0:
            messagebox.showwarning(
                "Warning",
                "No labels to save. Please load a video and label at least one frame.",
            )
            return

        # Ensure current frame labels are stored
        self._store_current_labels()

        default_name = "contact_labels.csv"
        if self.video_path is not None:
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            default_name = f"{base}_contacts.csv"

        csv_path = filedialog.asksaveasfilename(
            title="Save contact labels CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not csv_path:
            return

        # Save with a commented header so numpy.loadtxt can read it easily
        np.savetxt(
            csv_path,
            self.contact_labels,
            fmt="%d",
            delimiter=",",
            header="left_contact,right_contact",
            comments="# ",
        )

        messagebox.showinfo(
            "Saved",
            f"Contact labels saved to:\n{csv_path}\n\n"
            "Format: T x 2 CSV (left_contact, right_contact) with values 0 or 1.",
        )


def main() -> None:
    root = tk.Tk()
    app = ContactLabelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


