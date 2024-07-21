import customtkinter as ctk
from PIL import Image
from tkinter import filedialog
import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from DogVsWolves.components.model_manager import ConvolutionalNeuralNetwork
from DogVsWolves.config.configuration import ConfigurationManager

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog vs Cat Image Classifier")
        self.root.geometry("600x800")

        self.load_model()


        self.image_label = ctk.CTkLabel(root, width=540, height=300, 
                                        fg_color="white", text = "Image(Dog or Wolf)", text_color = "black", 
                                        font=("Helvetica",30))
        self.image_label.grid(row=0, column=0, columnspan=2, padx=30, pady=30)

        self.upload_button = ctk.CTkButton(root, width=540, height=50, 
                                           text="Upload Image", font=("Helvetica",22), command=self.upload_image)
        self.upload_button.grid(row=1, column=0, columnspan=2, padx=30, pady=10)

        self.classify_button = ctk.CTkButton(root, width=540, height=50, 
                                             text="Classify Image", font=("Helvetica",22), command=self.classify_image)
        self.classify_button.grid(row=2, column=0, columnspan=2, padx=30, pady=10)

        self.result_label = ctk.CTkLabel(root, text="", font=("Helvetica", 16))
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

        self.image_path = None


    def load_model(self):
        self.config = ConfigurationManager()
        self.config = self.config.get_evaluation_config()
        self.model = ConvolutionalNeuralNetwork()
        self.model.load_state_dict(torch.load(self.config.trained_model_inference_path))
        self.model.eval()


    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.img = ctk.CTkImage(dark_image=Image.open(self.image_path), size=(540, 300))
            self.image_label.configure(image=self.img)
            self.image_label.image = self.img
            self.image_label.configure(text="")

    def classify_image(self):
        if not self.image_path:
            self.result_label.configure(text="Please upload an image first.")

        image = read_image(self.image_path)
        image = self.transform_image(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image)
            print(outputs)
            probability = torch.softmax(outputs, dim = 1).squeeze()
            predicted_class = torch.argmax(probability)
            class_names = ["Wolf", "Dog"]
            result = class_names[predicted_class.item()]

        self.result_label.configure(text=f"Prediction: {result} (Probability: {round(probability[predicted_class].item(), 4)})")


    def transform_image(self, image):
        transform = v2.Compose([
            v2.Resize((self.config.params_image_size, self.config.params_image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Pytorch standard
        ])
        
        return transform(image)


if __name__ == "__main__":
    ctk.set_appearance_mode("System")  # Can change to "Light" or "Dark"
    ctk.set_default_color_theme("blue")  # Can change to "green" or "dark-blue"
    
    root = ctk.CTk()
    app = ImageApp(root)
    root.mainloop()
