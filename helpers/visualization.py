import glob
import os

import ipywidgets as widgets
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_colwidth', 500)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ipywidgets import interactive, widgets, Layout, Box
from IPython.display import display
from IPython.display import clear_output

from scipy.signal import savgol_filter


class ExperimentsBoard:
    def __init__(self, train_folder, value='val_loss'):
        self.train_folder = train_folder
        self.value = value
        self.create()

    def create(self):
        self.dfs, log_variables = self.read_logs()
        self.items_layout = Layout(flex='1 1 auto',
                                   width='auto',
                                   min_width='100px')

        box_layout = Layout(display='flex',
                            flex_flow='column',
                            align_items='stretch',
                            width='75%')

        refresh_button = widgets.Button(description='Refresh', layout=self.items_layout)
        refresh_button.on_click(self.on_refresh_button)
        self.metric_multiple = widgets.SelectMultiple(
            options=log_variables,
            value=[self.value],
            rows=3,
            description='metrics',
            disabled=False
        )
        interactive(self.update_plots, changevalue=self.metric_multiple)
        self.smoothing_factor = widgets.IntSlider(value=5, min=0, max=20, step=1,
                                                  description='Smoothing:', orientation='horizontal', readout=True,
                                                  readout_format='d', slider_color='white')
        self.smoothing_alpha = widgets.FloatSlider(value=0.1, min=0, max=1, step=0.1,
                                                   description='Smoothing Alpha:', orientation='horizontal',
                                                   readout=True, readout_format='d', slider_color='white')

        select_all = widgets.Button(description='Select All', layout=self.items_layout)
        select_all.on_click(self.on_select_all)
        select_none = widgets.Button(description='Select None', layout=self.items_layout)
        select_none.on_click(self.on_select_none)

        interactive(self.update_plots, changevalue=self.smoothing_factor)
        interactive(self.update_plots, changevalue=self.smoothing_alpha)
        items = [refresh_button, self.metric_multiple, self.smoothing_factor, self.smoothing_alpha,
                 widgets.HBox([select_all, select_none])]
        box1 = Box(children=items, layout=box_layout)
        self.output = widgets.Output(layout=box_layout)
        evbox = self.get_experiments()
        leftbox = widgets.VBox([box1, evbox])
        self.hbox = widgets.HBox([leftbox, self.output])

    def on_refresh_button(self, b):
        self.dfs, log_variables = self.read_logs()
        self.update_plots(b)

    def on_select_all(self, b):
        for key, df in self.dfs.items():
            self.big_dict[key]['checkbox'].value = True

    def on_select_none(self, b):
        for key, df in self.dfs.items():
            self.big_dict[key]['checkbox'].value = False

    def read_logs(self):
        dfs = dict()
        csvs = []
        for folder in sorted(os.listdir(self.train_folder)):
            csvs.extend(sorted(glob.glob(os.path.join(self.train_folder, folder) + '/*.csv')))
        for csv in csvs:
            if os.path.getsize(csv) > 0:
                dfs[os.path.dirname(csv).split('/')[-1]] = pd.read_csv(csv)
        log_variables = []
        for _, df in dfs.items():
            log_variables.extend(df.columns)
        log_variables = list(set(log_variables) - set(['epoch', ]))
        return dfs, sorted(log_variables)

    def get_experiments(self):
        experiments = [folder for folder in sorted(os.listdir(self.train_folder)) if 'board_logs' not in folder]
        pallete = sns.color_palette("Set1", n_colors=len(experiments)).as_hex()
        checkboxes = []
        colorpickers = []
        self.big_dict = dict()
        for color, folder in zip(pallete, experiments):
            folder_dict = dict()
            # checkbox_layout = Layout(min_width='100px')
            checkbox = widgets.ToggleButton(description=folder, value=False, layout=self.items_layout)
            checkboxes.append(checkbox)
            interactive(self.update_plots, changevalue=checkbox)
            colorpicker = widgets.ColorPicker(description='color', concise=True, value=color)
            colorpickers.append(colorpicker)
            interactive(self.update_plots, changevalue=colorpicker)
            folder_dict['colorpicker'] = colorpicker
            folder_dict['checkbox'] = checkbox
            self.big_dict[folder] = folder_dict
        box_layout1 = Layout(display='flex',
                             flex_flow='row',
                             align_items='stretch',
                             width='75%')
        boxes = []
        for chb, cp in zip(checkboxes, colorpickers):
            boxes.append(Box(children=[widgets.HBox([chb, cp])], layout=box_layout1))
        evbox = widgets.VBox(boxes)
        return evbox

    def update_plots(self, changevalue):
        current_metrics = self.metric_multiple.value
        with self.output:
            plt.figure(figsize=(12, 8), dpi=120);
            marker = None
            for i, metric in enumerate(current_metrics):
                for key, df in self.dfs.items():
                    if len(current_metrics) > 1:
                        marker = Line2D.filled_markers[i]
                    if self.big_dict[key]['checkbox'].value:
                        x = df['epoch'][:]
                        y = df[metric][:]
                        alpha = 1
                        color = self.big_dict[key]['colorpicker'].value
                        if 'loss' in metric:
                            label = key + ' ' + metric + ' min: {0:.4f}'.format((np.amin(df[metric])))
                            plt.plot(np.argmin(df[metric]), np.amin(df[metric]), marker='*', markersize=20, color=color)
                        else:
                            label = key + ' ' + metric + ' max: {0:.4f}'.format((np.amax(df[metric])))
                            plt.plot(np.argmax(df[metric]), np.amax(df[metric]), marker='*', markersize=20, color=color)
                        if self.smoothing_factor.value > 2:
                            alpha = self.smoothing_alpha.value
                            window_length = self.smoothing_factor.value * 2 + 1
                            if window_length < len(y):
                                y_smooth = savgol_filter(y, window_length, 2)
                            else:
                                y_smooth = y
                            plt.plot(x, y_smooth, label=label, color=color, marker=marker)
                            label = ''
                        plt.plot(x, y, alpha=alpha, color=color, label=label)
                        #                 plt.ylabel(metric)
            plt.xlabel('Epoch')
            plt.legend()
            clear_output(wait=True)
            plt.show()


class ExperimentsDetails:
    def __init__(self, train_folder):
        self.train_folder = train_folder
        self.create()

    def create(self):
        experiments = [folder for folder in sorted(os.listdir(self.train_folder)) if 'board_logs' not in folder]
        self.experiments_dropdown = widgets.Dropdown(options=experiments, value=experiments[0],
                                                     description='experiment: ')
        interactive(self.update_tables, experiment=self.experiments_dropdown)
        self.output = widgets.Output()
        self.vbox = widgets.VBox([self.experiments_dropdown, self.output])

    def get_chronicle(self, experiment):
        jsons = glob.glob(os.path.join(self.train_folder, experiment, 'chronicle') + '/*.json')
        dfs = []
        for json_file in jsons:
            df = pd.read_json(json_file)
            dfs.append(df)
        return (dfs)

    def update_tables(self, experiment):
        with self.output:
            dfs = self.get_chronicle(experiment)
            for df in dfs:
                display(df)
            clear_output(wait=True)


class Browse_image_label_2d_fcn_npy:
    def __init__(self, imgs, labels, preds=None, threshold=0.99):
        self.imgs = imgs
        self.labels = labels
        self.preds = preds
        self.threshold = threshold
        box_layout = Layout(align_items='stretch',
                            width='100%',
                            height='100%'
                            )
        image_slider = widgets.IntSlider(min=0, max=imgs.shape[0] - 1)
        interactive(self.view_image, sample=image_slider)
        self.output = widgets.Output(layout=box_layout)
        self.vbox = widgets.VBox([image_slider, self.output])

    def view_image(self, sample):
        with self.output:
            imgs = self.imgs
            labels = self.labels
            padx = int((imgs[0].shape[0] - labels[0].shape[0]) / 2)
            pady = int((imgs[0].shape[1] - labels[0].shape[1]) / 2)
            image = imgs[sample, :, :, 0]
            label = labels[sample, :, :, 0]
            label[label >= self.threshold] = 1
            label[label < self.threshold] = 0
            if self.preds is not None:
                fig, axs = plt.subplots(3, 1, figsize=(16, 18), facecolor='w', edgecolor='k')
            else:
                fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace=0.1, wspace=0.3)
            axs = axs.ravel()
            im1 = axs[0].imshow(image, cmap=plt.cm.gray, interpolation='none')
            axs[0].grid(False)
            divider1 = make_axes_locatable(axs[0])
            cax1 = divider1.append_axes("right", size="3%", pad=0.2)
            plt.colorbar(im1, cax=cax1, format="%.1f")
            #
            im2 = axs[1].imshow(image[padx:-padx, pady:-pady], cmap=plt.cm.gray, interpolation='none')
            divider2 = make_axes_locatable(axs[1])
            cax2 = divider2.append_axes("right", size="3%", pad=0.2)
            plt.colorbar(im2, cax=cax2, format="%.1f")
            mask = np.ma.masked_where(label == 0, label)
            axs[1].imshow(mask, cmap=plt.cm.jet, alpha=0.5)
            axs[1].grid(False)
            #
            if self.preds is not None:
                pred = self.preds[sample, :, :, 0]
                pred[pred >= self.threshold] = 1
                pred[pred < self.threshold] = 0
                im3 = axs[2].imshow(image[padx:-padx, pady:-pady], cmap=plt.cm.gray, interpolation='none')
                divider3 = make_axes_locatable(axs[2])
                cax3 = divider3.append_axes("right", size="3%", pad=0.2)
                plt.colorbar(im3, cax=cax3, format="%.1f")
                mask_pred = np.ma.masked_where(pred == 0, pred)
                axs[2].imshow(mask_pred, cmap=plt.cm.jet, alpha=0.5)
                axs[2].grid(False)
            clear_output(wait=True)
            plt.show()
