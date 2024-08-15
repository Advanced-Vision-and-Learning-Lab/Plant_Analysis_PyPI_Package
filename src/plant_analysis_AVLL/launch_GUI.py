from plant_analysis_AVLL import GUI

# launch GUI
def launch_GUI(model_path = 'yolo_segmentation_model.pt' , share_option = False):
    gui = GUI()
    gui.plant_analysis.load_segmentation_model(path = model_path)
    demo = gui.demo
    demo.launch(share = share_option)
