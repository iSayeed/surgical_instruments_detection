import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests
from pathlib import Path
import json
import os

class SurgicalToolsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Surgical Tools Detection")
        self.root.geometry("600x400")
        
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

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Set Type Selection
        ttk.Label(main_frame, text="Set Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.set_type = ttk.Combobox(main_frame, values=list(self.reference_data.keys()))
        self.set_type.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        if self.reference_data:
            self.set_type.set(list(self.reference_data.keys())[0])

        # Actual Weight
        ttk.Label(main_frame, text="Actual Weight:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.weight_var = tk.StringVar(value="0.0")
        self.weight_entry = ttk.Entry(main_frame, textvariable=self.weight_var)
        self.weight_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Image Selection
        ttk.Label(main_frame, text="Image:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.image_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.image_path, state='readonly').grid(
            row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(
            row=2, column=2, sticky=tk.W, pady=5)

        # Submit Button
        ttk.Button(main_frame, text="Detect Tools", command=self.submit).grid(
            row=3, column=0, columnspan=3, pady=20)

        # Results Area
        self.results_text = tk.Text(main_frame, height=10, width=50)
        self.results_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Configure grid
        main_frame.columnconfigure(1, weight=1)

    def browse_image(self):
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if filename:
            self.image_path.set(filename)

    def submit(self):
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
            'image': ('image.jpg', open(self.image_path.get(), 'rb'), 'image/jpeg')
        }
        data = {
            'set_type': self.set_type.get(),
            'actual_weight': weight
        }

        try:
            # Make the request
            response = requests.post(
                'http://127.0.0.1:8000/infer',
                files=files,
                data=data
            )
            response.raise_for_status()
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            result = response.json()
            
            # Format and display the results
            self.results_text.insert(tk.END, "Detection Results:\n\n")
            
            # Display detected instruments
            self.results_text.insert(tk.END, "Detected Instruments:\n")
            for instrument in result['detected_instruments']:
                self.results_text.insert(
                    tk.END, 
                    f"- {instrument['type']}: {instrument['count']}\n"
                )
            
            # Display set completion status
            self.results_text.insert(tk.END, f"\nSet Complete: {result['set_complete']}\n")
            
            # Display missing items if any
            if result['missing_items']:
                self.results_text.insert(tk.END, "\nMissing Items:\n")
                for item in result['missing_items']:
                    self.results_text.insert(
                        tk.END,
                        f"- {item['type']}: Found {item['found']}, Expected {item['expected']}\n"
                    )

        except requests.RequestException as e:
            messagebox.showerror("Error", f"Request failed: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = SurgicalToolsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
