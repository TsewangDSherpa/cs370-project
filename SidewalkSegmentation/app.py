import os
import shutil
from shiny.express import input, render, ui
from functools import partial
from shiny.ui import page_navbar
from htmltools import div, Tag
from SamPredict import Predict
import requests

# Define the directory where the uploaded images will be stored
uploaded_dir = "./UploadedImg"

# Ensure the directory exists
os.makedirs(uploaded_dir, exist_ok=True)
os.makedirs("./Result", exist_ok=True)

ui.page_opts(
    title="Finetuned Segmentation Model for Sidewalks",
    page_fn=partial(page_navbar, id="page"),
)

with ui.nav_panel("Segment Sidewalk"):
    ui.h4(
        "Input an image of a sidewalk from satellite, and segment the sidewalk from it.",
        style="text-align: center;",
    )
    ui.hr()
    html_string = '<div style="display: flex; flex-direction:column; align-items: center; justify-content: center; text-align: center; height:100vh; width:100vw;">'

    ui.HTML(html_string)
  
    ui.input_file("upload", "Upload Sidewalk Tile Image", multiple=False, accept=[".jpeg", ".png"])
    
    
    

    
    @render.ui
    def render_fig0Open():
        InputFile = input.upload()
        if InputFile:
            Data = InputFile[0]
            if Data["type"] in ["image/jpeg", "image/png"]:
                # Copy the uploaded image to the UploadedImg directory
                uploaded_path = os.path.join(uploaded_dir, Data["name"])
                shutil.copy(Data["datapath"], uploaded_path)
                # Construct the image source with the path relative to the server
                image_src = f"./UploadedImg/{Data['name']}"
                print(f"Image loading from: {image_src}")
                return [ui.HTML('<figure style="border: 1px #cccccc solid; padding: 4px; box-shadow: 2px 3px 8px 4px #888888">'),  ui.HTML('<figcaption style="text-align:center; padding:10px;">Original Input Image</figcaption></figure>')]
            else:
                print("Enter a proper JPEG/PNG file!")
                return ui.h1("Enter a proper JPEG/PNG file!")
            
    
    @render.image(delete_file=True)
    def image():
        InputFile = input.upload()
        if InputFile:
            Data = InputFile[0]
            if Data["type"] in ["image/jpeg", "image/png"]:
                # Construct the image source with the path relative to the server
                image_src = f"./UploadedImg/{Data['name']}"
                print(f"Image loading from: {image_src}")
                value = {"src": image_src, "width": "400px", "height": "335px"}
                return value
            
    @render.ui
    def render_fig1Open():
        InputFile = input.upload()
        if InputFile:
            Data = InputFile[0]
            if Data["type"] in ["image/jpeg", "image/png"]:
                # Copy the uploaded image to the UploadedImg directory
                uploaded_path = os.path.join(uploaded_dir, Data["name"])
                shutil.copy(Data["datapath"], uploaded_path)
                # Construct the image source with the path relative to the server
                image_src = f"./UploadedImg/{Data['name']}"
                print(f"Image loading from: {image_src}")
                return [ui.HTML('<figure style="border: 1px #cccccc solid; padding: 4px; box-shadow: 2px 3px 8px 4px #888888">'),  ui.HTML('<figcaption style="text-align:center; padding:10px;">SideWalk Segmentation Image</figcaption></figure>')]
            else:
                print("Enter a proper JPEG/PNG file!")
               
        
    
    
    @render.image(delete_file=True)
    def result_img():
        InputFile = input.upload()
        if InputFile:
            Data = InputFile[0]
            if Data["type"] in ["image/jpeg", "image/png"]:
                # Construct the image source with the path relative to the server
                output_src = Predict(Data["name"])
                print(f"Image loading from: {output_src}")
                os.remove(f"./UploadedImg/{Data['name']}")
                return {"src": output_src, "width": "400px", "height": "335px"}



    ui.HTML("</div>")

with ui.nav_panel("About Us"):
    ui.h2("CS 370 Project")
    ui.p("This project is done by Norsang Nyandak and Tsewang Sherpa")
    ui.p(
        "Group Project that makes use of SAM (Segment Anything Model) to train on custom dataset of sidewalks, using which sidewalks could be extracted from the given input Image."
    )
