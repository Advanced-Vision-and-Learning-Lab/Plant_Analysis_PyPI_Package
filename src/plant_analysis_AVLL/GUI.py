from Plant_Analysis import Plant_Analysis
import gradio as gr
from time import time

class GUI():

    def __init__(self, device = 'cpu'):

        self.session_index = 1
        self.device = device

        self.head = (
                    "<center>"
                    "<a href='https://precisiongreenhouse.tamu.edu/'><img src='https://peepleslab.engr.tamu.edu/wp-content/uploads/sites/268/2023/04/AgriLife_Logo-e1681857158121.png' width=1650></a>"
                    "<br>"
                    "Plant Analysis and Feature Extraction Demonstration"
                    "<br>"
                    "<a href ='https://precisiongreenhouse.tamu.edu/'>The Texas A&M Plant Growth and Phenotyping Facility Data Analysis Pipeline</a>"
                    "</center>"
                )
        

        self.theme = gr.themes.Base(
                primary_hue="violet",
                secondary_hue="green",).set(body_background_fill_dark='*checkbox_label_background_fill')
        self.demo = gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.lime))
        self.service_dropdown_choices = ['Multi Plant Analysis', 'Single Plant Analysis']
        self.plant_analysis = {}
        
        with self.demo:

            self.session_name = gr.State([])
            
            gr.HTML(value = self.head)
            
            with gr.Column():
                
                self.service_dropdown = gr.Dropdown(choices = self.service_dropdown_choices, 
                                       multiselect = False, 
                                       label = 'Select Service',
                                       show_label = True, 
                                       visible = True,
                                       type = 'index')
                
                self.filepath_input = gr.Textbox(label = 'Enter folder path containing plant images',
                                            show_label = True,
                                            type = 'text',
                                            visible = False)

                self.input_submit_button = gr.Button(value = 'Submit Input Folder Path',
                                          visible = False)
                
                with gr.Row():
                    self.show_input_checkbox = gr.Checkbox(label = 'Show Raw Images',
                                                      info = 'Check to show input images',
                                                      value = False,
                                                      visible = False)
                    
                    self.show_color_images_checkbox = gr.Checkbox(label = 'Show Color Images',
                                                      info = 'Check to show color images',
                                                      value = False,
                                                      visible = False)
                    
                self.request_submit_button = gr.Button(value = 'Submit',
                                                       visible = False)
                
                self.pre_information_textbox = gr.Textbox(label = 'Information',
                                                        visible = False)

                self.post_information_textbox = gr.Textbox(label = 'Information',
                                                        visible = False)                

                #outputs

                with gr.Tabs():
                    
                    self.plant_height_tab = gr.Tab(label = 'Height', visible = False)
                    self.plant_width_tab = gr.Tab(label = 'Width', visible = False)
                    self.plant_area_tab = gr.Tab(label = 'Area', visible = False)
                    
                    with self.plant_height_tab:
                        
                        self.plant_height_plot = gr.BarPlot(visible = False)
                        
                    with self.plant_width_tab:
                        
                        self.plant_width_plot = gr.BarPlot(visible = False)
                    
                    with self.plant_area_tab:
                        
                        self.plant_area_plot = gr.BarPlot(visible = False)
                    
                
                self.plant_select_dropdown = gr.Dropdown( multiselect = False, 
                                                       label = 'Select Plant',
                                                       show_label = True, 
                                                       visible = False,
                                                       type = 'value')
                
                with gr.Tabs():
                    
                    self.input_images_tab = gr.Tab(label = 'Raw Input Images', visible = False)
                    self.color_images_tab = gr.Tab(label = 'Color Input Images', visible = False)
                    self.plant_analysis_tab = gr.Tab(label = 'Plant Analysis', visible = False)
                    self.plant_statistics_tab = gr.Tab(label = 'Plant Statistics', visible = False)
                    
                    with self.input_images_tab:
                        
                        self.input_images_gallery = gr.Gallery(label = 'Uploaded Raw Input Images',
                                                    show_label = True,
                                                    height = 512,
                                                    preview = True,
                                                    visible = False)
                    with self.color_images_tab:
                        
                        self.color_images_gallery = gr.Gallery(label = 'Color Input Images',
                                                    show_label = True,
                                                    height = 512,
                                                    preview = True,
                                                    visible = False)
                    
                    with self.plant_analysis_tab:
                        
                        self.plant_analysis_gallery = gr.Gallery(label = 'Plant Analysis',
                                                    show_label = True,
                                                    height = 512,
                                                    preview = True,
                                                    visible = False)
                    
                    with self.plant_statistics_tab:
                        
                        self.plant_statistics_df = gr.Dataframe(label = 'Plant Phenotypes',
                                             show_label = True,
                                             visible = False)

                self.output_folder_textbox = gr.Textbox(label = 'Enter path to save results to',
                                                       show_label = True,
                                                       visible = False)

                self.save_result_button = gr.Button(value = 'SAVE RESULTS',
                                                   visible = False)

                self.saving_information_textbox = gr.Textbox(label = 'Information',
                                                            show_label = False,
                                                            visible = False)

                self.reset_button = gr.ClearButton(components = [self.filepath_input,
                                                self.show_input_checkbox,
                                                self.show_color_images_checkbox,
                                                self.pre_information_textbox,
                                                self.post_information_textbox,
                                                self.plant_select_dropdown,
                                                self.input_images_tab,
                                                self.input_images_gallery,
                                                self.color_images_tab,
                                                self.color_images_gallery,
                                                self.plant_analysis_tab,
                                                self.plant_analysis_gallery,
                                                self.plant_statistics_tab,
                                                self.plant_statistics_df,
                                                self.output_folder_textbox,
                                                self.saving_information_textbox],
                                                value = 'CLEAR',
                                                visible = False)
                    
        
            self.service_dropdown.input(self.update_service,
                                        inputs = [self.session_name,
                                                  self.service_dropdown],
                                        outputs = [self.filepath_input,
                                                   self.input_submit_button,
                                                   self.session_name])

            self.input_submit_button.click(self.update_input_path,
                                           inputs = [self.session_name,
                                                    self.filepath_input],
                                           outputs = [self.show_input_checkbox,
                                                      self.show_color_images_checkbox,
                                                      self.request_submit_button])

            self.show_input_checkbox.input(self.update_check_RI_option,
                                           inputs = [self.session_name,
                                                    self.show_input_checkbox])

            self.show_color_images_checkbox.input(self.update_check_CI_option,
                                                  inputs = [self.session_name,
                                                            self.show_color_images_checkbox])
            
            self.request_submit_button.click(self.update_info_textbox,
                                             inputs = self.session_name,
                                             outputs = self.pre_information_textbox)

            self.request_submit_button.click(self.get_plant_analysis,
                                             inputs = self.session_name,
                                             outputs = [self.post_information_textbox,
                                                        self.plant_height_tab,
                                                        self.plant_width_tab,
                                                        self.plant_area_tab,
                                                        self.plant_height_plot,
                                                        self.plant_width_plot,
                                                        self.plant_area_plot,
                                                        self.plant_select_dropdown])
            
            self.plant_select_dropdown.input(self.show_plant_analysis_result,
                                             inputs = [self.session_name,
                                                       self.plant_select_dropdown],
                                             outputs = [self.input_images_tab, 
                                                        self.color_images_tab, 
                                                        self.plant_analysis_tab, 
                                                        self.plant_statistics_tab,
                                                        self.input_images_gallery,
                                                        self.color_images_gallery,
                                                        self.plant_analysis_gallery,
                                                        self.plant_statistics_df,
                                                        self.output_folder_textbox,
                                                        self.save_result_button,
                                                        self.reset_button])

            self.save_result_button.click(self.save_analysis_result,
                                         inputs = [self.session_name,
                                                   self.output_folder_textbox],
                                         outputs = self.saving_information_textbox)
            
            self.reset_button.click(self.reset, inputs = self.session_name,  js="window.location.reload()")
            
    
    def update_service(self, session, service_type):
        
        session.append('session_'+str(self.session_index))
        self.plant_analysis[session[0]] = Plant_Analysis(device = self.device)
        self.session_index += 1
        if self.session_index == 100000:
            self.session_index = 1

        self.plant_analysis[session[0]].update_service_type(service_type)
        outputs = []
        outputs.append(gr.Textbox(label = 'Enter folder path containing plant images',
                                            show_label = True,
                                            type = 'text',
                                            visible = True))
        outputs.append(gr.Button(value = 'Submit Input Folder Path',
                                          visible = True))
        outputs.append(session)
        return outputs
        
    def update_input_path(self, session, input_path):

        self.plant_analysis[session[0]].update_input_path(input_path)
        outputs = []
        outputs.append(gr.Checkbox(label = 'Show Raw Images',
                                                      info = 'Check to show input images',
                                                      value = False,
                                                      visible = True))
        outputs.append(gr.Checkbox(label = 'Show Color Images',
                                                      info = 'Check to show color images',
                                                      value = False,
                                                      visible = True))
        outputs.append(gr.Button(value = 'Submit',
                                          visible = True))
        return outputs

    def update_check_RI_option(self, session,  check_RI):

        self.plant_analysis[session[0]].update_check_RI_option(check_RI)

    def update_check_CI_option(self, session, check_CI):

        self.plant_analysis[session[0]].update_check_CI_option(check_CI)

    def update_info_textbox(self, session):

        information = 'Request Submitted. Processing ' + str(len(list(self.plant_analysis[session[0]].plants.keys()))) + ' Plants.\n\nPlease wait..'

        return gr.Textbox(show_label = False,
                          value = information,
                          visible = True)

    def get_plant_analysis(self, session):

        self.plant_analysis[session[0]].make_color_images()
        self.plant_analysis[session[0]].stitch_color_images()
        self.plant_analysis[session[0]].calculate_connected_components()
        self.plant_analysis[session[0]].load_segmentation_model()
        self.plant_analysis[session[0]].run_segmentation()
        self.plant_analysis[session[0]].calculate_plant_phenotypes()
        self.plant_analysis[session[0]].calculate_tips_and_branches()
        self.plant_analysis[session[0]].calculate_sift_features()
        self.plant_analysis[session[0]].calculate_LBP_features()
        self.plant_analysis[session[0]].calculate_HOG_features()
        information = 'Processing Complete.\n\nSelect plant from the dropdown below to check the output of individual plants'

        outputs = []

        outputs.append(gr.Textbox(show_label = False,
                                  value = information,
                                  visible = True))
        outputs.append(gr.Tab(label = 'Height', visible = True))
        outputs.append(gr.Tab(label = 'Width', visible = True))
        outputs.append(gr.Tab(label = 'Area', visible = True))
        plant_statistics_df = self.plant_analysis[session[0]].get_plant_statistics_df()
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Height',
                                  title = 'Plant Height',
                                  tooltip = 'Height',
                                  x_title = 'Plant',
                                  y_title = 'Height (cm)',
                                  x_label_angle = -45,
                                  y_label_angle = 0,
                                  label = 'Plant Height Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Width',
                                  title = 'Plant Width',
                                  tooltip = 'Width',
                                  x_title = 'Plant',
                                  y_title = 'Width (cm)',
                                  x_label_angle = -45,
                                  y_label_angle = 0,
                                  label = 'Plant Width Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.BarPlot(value = plant_statistics_df,
                                  x = 'Plant_Name',
                                  y = 'Area',
                                  title = 'Plant Area',
                                  tooltip = 'Area',
                                  x_title = 'Plant',
                                  y_title = 'Area (square cm)',
                                  x_label_angle = -45,
                                  y_label_angle = 0,
                                  label = 'Plant Area Plot',
                                  show_label = True,
                                  visible = True))
        outputs.append(gr.Dropdown(choices = self.plant_analysis[session[0]].get_plant_names(),
                           multiselect = False, 
                           label = 'Select Plant',
                           show_label = True, 
                           visible = True,
                           type = 'value'))

        return outputs

    def show_plant_analysis_result(self, session, plant):

        outputs = []

        outputs.append(gr.Tab(label = 'Raw Input Images', visible = self.plant_analysis[session[0]].show_raw_images))
        outputs.append(gr.Tab(label = 'Color Input Images', visible = self.plant_analysis[session[0]].show_color_images))
        outputs.append(gr.Tab(label = 'Plant Analysis', visible = True))
        outputs.append(gr.Tab(label = 'Plant Statistics', visible = True))
        outputs.append(gr.Gallery(value = self.plant_analysis[session[0]].get_raw_images(plant) if self.plant_analysis[session[0]].show_raw_images else [],
                                label = 'Uploaded Raw Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = self.plant_analysis[session[0]].show_raw_images))
        outputs.append(gr.Gallery(value = self.plant_analysis[session[0]].get_color_images(plant) if self.plant_analysis[session[0]].show_color_images else [],
                                label = 'Color Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = self.plant_analysis[session[0]].show_color_images))
        outputs.append(gr.Gallery(value = self.plant_analysis[session[0]].get_plant_analysis_images(plant),
                                label = 'Plant Analysis',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = True))
        outputs.append(gr.Textbox(value = self.plant_analysis[session[0]].get_plant_statistics_df_plantwise(plant),
                                 label = 'Estimated Plant Height is ',
                                 show_label = True,
                                 visible = True))
        outputs.append(gr.Textbox(label = 'Enter path to save results to',
                               show_label = True,
                               value = 'Results',
                               visible = True))
        outputs.append(gr.Button(value = 'SAVE RESULTS',
                               visible = True))
        outputs.append(gr.ClearButton(components = [self.filepath_input,
                                                self.show_input_checkbox,
                                                self.show_color_images_checkbox,
                                                self.pre_information_textbox,
                                                self.post_information_textbox,
                                                self.plant_select_dropdown,
                                                self.input_images_tab,
                                                self.input_images_gallery,
                                                self.color_images_tab,
                                                self.color_images_gallery,
                                                self.plant_analysis_tab,
                                                self.plant_analysis_gallery,
                                                self.plant_statistics_tab,
                                                self.plant_statistics_df,
                                                self.output_folder_textbox,
                                                self.saving_information_textbox],
                                                value = 'CLEAR',
                                                visible = True))

        return outputs

    def save_analysis_result(self, session, output_folder_path):

        self.plant_analysis[session[0]].save_results(output_folder_path)

        return gr.Textbox(label = 'Information',
                         show_label = False,
                         value = 'Saved Result',
                         visible = True)
    
    def reset(self, session):

        self.plant_analysis[session[0]].reset()
        del self.plant_analysis[session[0]]
        print(session[0] + ' is cleared')

    def reset_depricated(self):

        self.plant_analysis.reset()
        
        outputs = []
        
        outputs.append(gr.Textbox(label = 'Enter folder path containing plant images',
                                            show_label = True,
                                            type = 'text',
                                            visible = False))
        outputs.append(gr.Button(value = 'Submit Input Folder Path',
                                          visible = False))
        outputs.append(gr.Checkbox(label = 'Show Raw Images',
                                                      info = 'Check to show input images',
                                                      value = False,
                                                      visible = False))
        outputs.append(gr.Checkbox(label = 'Show Color Images',
                                                      info = 'Check to show color images',
                                                      value = False,
                                                      visible = False))
        outputs.append(gr.Button(value = 'Submit',
                                          visible = False))
        outputs.append(gr.Dropdown( multiselect = False, 
                                   label = 'Select Plant',
                                   show_label = True, 
                                   visible = False,
                                   type = 'value'))
        outputs.append(gr.Tab(label = 'Raw Input Images', visible = False))
        outputs.append(gr.Tab(label = 'Color Input Images', visible = False))
        outputs.append(gr.Tab(label = 'Plant Analysis', visible = False))
        outputs.append(gr.Tab(label = 'Plant Statistics', visible = False))
        outputs.append(gr.Gallery(label = 'Uploaded Raw Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = False))
        outputs.append(gr.Gallery(label = 'Color Input Images',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = False))
        outputs.append(gr.Gallery(label = 'Plant Analysis',
                                show_label = True,
                                height = 512,
                                preview = True,
                                visible = False))
        outputs.append(gr.Textbox(label = 'Estimated Plant Height is ',
                                 show_label = True,
                                 visible = False))
        outputs.append(gr.Button(value = 'CLEAR',
                              visible = False))

        return outputs
