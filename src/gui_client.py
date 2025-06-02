from __future__ import annotations

import sys
import json
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import requests
from PIL import Image  # PIL for image processing
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


class SurgicalToolsGUI:
    """A GUI application for detecting surgical tools in images and validating surgical tool sets."""

    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the GUI application.

        Args:
            root: The root window of the application.

        """
        self.root = root
        self.root.title("Surgical Tools Detection")
        self.root.geometry("1200x600")  # Made window larger for better image viewing
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.current_image = None  # Store the current PhotoImage

        # Load config to get set types
        try:
            config_path = Path(__file__).parent.parent / "config.json"
            with open(config_path) as f:
                config = json.load(f)
                self.reference_data = config["REFERENCE_DATA"]
        except Exception as e:
            messagebox.showerror("Error", f"Could not load config.json: {str(e)}")
            self.reference_data = {}

        self.create_widgets()

    def create_widgets(self) -> None:
        """Create and configure all GUI widgets and layout for the application."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status Panel frame (right side)
        status_frame = ttk.Frame(self.root, padding="10", relief="ridge", borderwidth=1)
        status_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Status display
        ttk.Label(
            status_frame, text="Missing Items", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.missing_count_label = ttk.Label(
            status_frame, text="0", font=("Helvetica", 36, "bold"), foreground="red")
        self.missing_count_label.pack(pady=10)

        # Make main_frame expandable
        main_frame.rowconfigure(4, weight=1)  # Make row with results text expandable
        main_frame.columnconfigure(1, weight=1)  # Make middle column expandable

        # Configure text tags for colors
        self.results_text = tk.Text(main_frame, height=10, width=50, wrap=tk.WORD)
        self.results_text.tag_configure("red", foreground="red")
        self.results_text.grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Add scrollbar to results text
        scrollbar = ttk.Scrollbar(
            main_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=4, column=3, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)

        # Set Type Selection
        ttk.Label(main_frame, text="Set Type:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.set_type = ttk.Combobox(
            main_frame, values=list(self.reference_data.keys()))
        self.set_type.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        if self.reference_data:
            self.set_type.set(next(iter(self.reference_data.keys())))

        # Actual Weight
        ttk.Label(main_frame, text="Actual Weight:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.weight_var = tk.StringVar(value="0.0")
        self.weight_entry = ttk.Entry(main_frame, textvariable=self.weight_var)
        self.weight_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Image Selection
        ttk.Label(main_frame, text="Image:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.image_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.image_path, state="readonly").grid(
            row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(
            row=2, column=2, sticky=tk.W, pady=5)

        # Submit Button
        ttk.Button(main_frame, text="Detect Tools", command=self.submit).grid(
            row=3, column=0, columnspan=3, pady=20)

        # Grid configuration is already handled at the top of create_widgets

    def browse_image(self) -> None:
        """Open a file dialog for selecting an image file and update the image path in the GUI."""
        filename = filedialog.askopenfilename(
            title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filename:
            self.image_path.set(filename)

    def display_image(self, image_path: str | Path) -> None:
        """
        Display an image in a new window with appropriate scaling.

        Args:
            image_path: Path to the image file as string or Path object.


        """
        try:
            logger.info(f"Loading image from: {image_path}")

            # Verify image path
            image_path = str(image_path)  # Convert Path to string if needed
            if not Path(image_path).exists():
                raise ValueError(f"Image file does not exist: {image_path}")

            # Create a new Toplevel window for the image
            image_window = tk.Toplevel(self.root)
            image_window.title("Detection Results")

            # Create a label to display the image
            image_label = ttk.Label(image_window)
            image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

            # Open and process image
            image = Image.open(image_path)
            logger.info(f"Original image size: {image.size}")

            # Convert to RGB if needed
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

            # Get screen dimensions
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Calculate maximum dimensions (80% of screen size)
            max_width = int(screen_width * 0.8)
            max_height = int(screen_height * 0.8)

            # Get original dimensions
            img_width, img_height = image.size

            # Calculate scaling ratio
            width_ratio = max_width / img_width
            height_ratio = max_height / img_height
            ratio = min(width_ratio, height_ratio)

            # Calculate new dimensions
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            logger.info(f"Resizing to: {new_width}x{new_height}")

            # Resize image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to temporary PNG file (PNG works better with Tkinter)
            temp_path = Path(image_path).parent / "temp_display.png"
            image.save(temp_path, format="PNG")

            # Create PhotoImage directly from file
            self.current_image = tk.PhotoImage(file=str(temp_path))
            logger.info("Created PhotoImage")

            # Display image in label
            image_label.configure(image=self.current_image)

            # Set window size and position
            window_width = new_width + 20  # Add padding
            window_height = new_height + 20
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            image_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

            logger.info("Image displayed in new window")

            # Clean up temporary file
            temp_path.unlink()

        except Exception as e:
            logger.error("Error displaying image:")
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def submit(self) -> None:
        """Handle the submission of the form, validate inputs, and send the detection request to the server."""
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image")
            return

        try:
            weight = float(self.weight_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid weight")
            return

        # Prepare the files and data for the request
        files = {
            "image": ("image.jpg", open(self.image_path.get(), "rb"), "image/jpeg"),
        }
        data = {"set_type": self.set_type.get(), "actual_weight": weight}

        try:
            # Make the request
            response = requests.post(
                "http://127.0.0.1:8000/infer", files=files, data=data, timeout=30)
            response.raise_for_status()

            # Display results
            self.results_text.delete(1.0, tk.END)
            result = response.json()
            logger.debug("Received response from server: {}", json.dumps(result, indent=2))

            # Format and display the results
            self.results_text.insert(tk.END, "Detection Results:\n\n")

            # Display detected instruments
            self.results_text.insert(tk.END, "Detected Instruments:\n")
            for instrument in result["detected_instruments"]:
                self.results_text.insert(
                    tk.END, f"- {instrument['type']}: {instrument['count']}\n",
                )

            # Display set completion status
            self.results_text.insert(
                tk.END, f"\nSet Complete: {result['set_complete']}\n",
            )

            # Update missing items count and display
            total_missing = len(result.get("missing_items", []))
            self.missing_count_label.configure(text=str(total_missing))

            # Display missing items if any
            if result["missing_items"]:
                self.results_text.insert(tk.END, "\nMissing Items ", "red")
                self.results_text.insert(tk.END, f"(Total: {total_missing}):\n", "red")
                for item in result["missing_items"]:
                    self.results_text.insert(
                        tk.END,
                        f"- {item['type']}: Found {item['found']}, Expected {item['expected']}\n",
                        "red",
                    )

            # Display the detected image and switch to the image tab
            if "predicted_image_path" in result:
                predicted_image_path = result["predicted_image_path"]
                try:
                    logger.info(f"Attempting to display image from: {predicted_image_path}")
                    if Path(predicted_image_path).is_file():
                        self.display_image(predicted_image_path)
                    else:
                        messagebox.showerror(
                            "Error",
                            f"Predicted image not found at: {predicted_image_path}",
                        )
                except Exception as e:
                    messagebox.showerror("Error", f"Error displaying image: {str(e)}")
            else:
                messagebox.showerror("Error", "No detection image received from server")

        except requests.RequestException as e:
            messagebox.showerror("Error", f"Request failed: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


def main() -> None:
    """Main entry point of the application that initializes and starts the GUI."""
    root = tk.Tk()
    SurgicalToolsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
