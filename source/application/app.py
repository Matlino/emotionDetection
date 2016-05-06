from time import sleep
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import GObject
GObject.threads_init()
from pylab import figure, setp
from numpy import arange
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from source.application import eeg_loger
import pandas as pd
from source.application import emotion_detection
from os.path import basename



happy_image_path = '..\\..\\app_pictures\\happy.png'
sad_image_path = '..\\..\\app_pictures\\sad.png'
disgusted_image_path = '..\\..\\app_pictures\\disgusted.png'
neutral_image_path = '..\\..\\app_pictures\\neutral.png'
scared_image_path = '..\\..\\app_pictures\\scared.png'
surprised_image_path = '..\\..\\app_pictures\\surprised.png'
angry_image_path = '..\\..\\app_pictures\\angry.png'
blank_image_path = '..\\..\\app_pictures\\blank.png'




class GridWindow(Gtk.Window):
    """
    This class represents GUI for application. All the button, labels and other widgets are displayed in init method.
    """
    def __init__(self):
        Gtk.Window.__init__(self, title="Emotion detection using EPOC EEG device")

        self.grid = Gtk.Grid()
        self.add(self.grid)

        # left main horizontal box
        self.left_horizontal_box = Gtk.Box(spacing=20, orientation=Gtk.Orientation.HORIZONTAL)
        self.grid.attach(self.left_horizontal_box, 0, 0, 1, 1)

        # right main horizontal box
        self.right_horizontal_box = Gtk.Box(spacing=20, orientation=Gtk.Orientation.HORIZONTAL)
        self.grid.attach(self.right_horizontal_box, 1, 0, 1, 1)

        # box for buttons
        self.vertical_box_left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.left_horizontal_box.pack_start(self.vertical_box_left, True, True, padding=20)

        # boxes for label and entry
        self.neweeg_box = Gtk.Box(spacing=20, orientation=Gtk.Orientation.HORIZONTAL)
        self.importeeg_box = Gtk.Box(spacing=20, orientation=Gtk.Orientation.HORIZONTAL)

        #arousa and valence boxes
        self.arousal_box = Gtk.Box(spacing=20, orientation=Gtk.Orientation.HORIZONTAL)
        self.valence_box = Gtk.Box(spacing=20, orientation=Gtk.Orientation.HORIZONTAL)

        # box for eeg siglal title
        # self.eegplot_title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        # self.right_horizontal_box.pack_start(self.eegplot_title_box, True, True, padding=20)

        # box for eeg plot
        self.eegplot_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.right_horizontal_box.pack_start(self.eegplot_box, True, True, padding=20)

        # create some buttons
        self.recording_button = Gtk.Button(label="Start recording")
        self.recording_button.connect("clicked", self.on_start_recording)

        self.import_button = Gtk.Button(label="Import EEG file")
        self.import_button.connect("clicked", self.on_file_choose)

        self.emotion_detect_button = Gtk.Button(label="Detect emotion")
        self.emotion_detect_button.connect("clicked", self.on_detect_emotion)

        #labels
        self.arousal_label = Gtk.Label(label="Arousal value: ")
        self.valence_label = Gtk.Label(label="Valence value: ")
        self.imported_file_label = Gtk.Label(label="Imported EEG file: ")
        self.new_file_label = Gtk.Label(label="New EEG filename: ")
        self.imported_filename_label = Gtk.Label()
        self.arousal_value_label = Gtk.Label()
        self.valence_value_label = Gtk.Label()

        self.create_eeg_label = Gtk.Label()
        self.create_eeg_label.set_markup("<b><big>Create new EEG file</big></b>")

        self.import_eeg_label = Gtk.Label()
        self.import_eeg_label.set_markup("<b><big>Import existing EEG file</big></b>")

        self.detect_emotion_label = Gtk.Label()
        self.detect_emotion_label.set_markup("<b><big>Detect emotion from EEG signal</big></b>")

        self.eegsignal_label = Gtk.Label()
        self.eegsignal_label.set_markup("<b><big>EEG signal</big></b>")

        #entries
        self.new_file_entry = Gtk.Entry()

        # emotion image
        self.image = Gtk.Image()
        self.image.set_from_file(blank_image_path)

        # add widgets to GUI
        self.vertical_box_left.pack_start(self.create_eeg_label, False, False, padding=10)
        self.vertical_box_left.pack_start(self.neweeg_box, False, False, 0)
        self.neweeg_box.pack_start(self.new_file_label, False, False, 0)
        self.neweeg_box.pack_start(self.new_file_entry, False, False, 0)
        self.vertical_box_left.pack_start(self.recording_button, False, False, padding=10)

        self.vertical_box_left.pack_start(self.import_eeg_label, False, False, padding=10)
        self.vertical_box_left.pack_start(self.import_button, True, True, 0)
        self.vertical_box_left.pack_start(self.importeeg_box, False, False, padding=10)
        self.importeeg_box.pack_start(self.imported_file_label, False, False, 0)
        self.importeeg_box.pack_start(self.imported_filename_label, False, False, 0)

        self.vertical_box_left.pack_start(self.detect_emotion_label, False, False, 0)
        self.vertical_box_left.pack_start(self.emotion_detect_button, False, False, padding=10)
        self.vertical_box_left.pack_start(self.arousal_box, False, False, 0)
        self.vertical_box_left.pack_start(self.valence_box, False, False, 0)
        self.arousal_box.pack_start(self.arousal_label, False, False, 0)
        self.arousal_box.pack_start(self.arousal_value_label, False, False, 0)
        self.valence_box.pack_start(self.valence_label, False, False, 0)
        self.valence_box.pack_start(self.valence_value_label, False, False, 0)
        self.vertical_box_left.pack_start(self.image, False, False, padding=20)


        # path to eeg fle
        self.eeg_file_path = ""

        # eeg plot setup
        self.eegplot_box.set_size_request(600, 300)
        self.fig = figure()
        self.clear_eeg_plot()
        self.canvas = FigureCanvas(self.fig)  # a gtk.DrawingArea

        self.eegplot_box.pack_start(self.eegsignal_label, False, False, padding=10)
        self.eegplot_box.pack_start(self.canvas, True, True, padding=20)

    def normalize_signal(self, signal):
        """Normalize signal so it can be displayed in graph.

        Parameters
        ----------
        signal: array, shape = [n_samples, 1]
                Eeg signal.

        Returns
        -------
        Normalized EEG signal
        """
        max_signal = max(signal)
        min_signal = min(signal)
        return (signal - min_signal) / (max_signal - min_signal)

    def clear_eeg_plot(self):
        """ Clear the plot so it can be updated.

        """
        self.fig.clf()

        self.yprops = dict(rotation=0,
                      horizontalalignment='right',
                      verticalalignment='center',
                      x=-0.01)

        self.axprops = dict(yticks=[])
        self.plot_color = 'lightblue'

        self.ax1 =self.fig.add_axes([0.1, 0.7, 0.8, 0.2], **self.axprops)

        self.axprops['sharex'] = self.ax1
        self.axprops['sharey'] = self.ax1
        # force x axes to remain in register, even with toolbar navigation
        self.ax2 = self.fig.add_axes([0.1, 0.5, 0.8, 0.2], **self.axprops)
        self.ax3 = self.fig.add_axes([0.1, 0.3, 0.8, 0.2], **self.axprops)
        self.ax4 = self.fig.add_axes([0.1, 0.1, 0.8, 0.2], **self.axprops)

        for ax in self.ax1, self.ax2, self.ax3, self.ax4:
            setp(ax.get_xticklabels(), visible=False)


    def on_start_recording(self, widget):
        """ Start recording EEG signal, Emotiv EPOC needs to be connected.

        """
        if eeg_loger.is_recording == 1:
            self.recording_button.set_label("Start recording")
            eeg_loger.is_recording = 0

            self.import_button.set_sensitive(True)
            self.emotion_detect_button.set_sensitive(True)

            # update graph
            # self.update_eeg_plot()
            #### GObject.idle_add(self.update_eeg_plot)

            dialog = DialogExample(self, "Information", "End of recording!")
            dialog.run()
            dialog.destroy()
        else:
            if self.new_file_entry.get_text() == "":
                dialog = DialogExample(self, "Error", "U did not set a file name!")
                dialog.run()
                dialog.destroy()
            else:
                self.recording_button.set_label("Stop recording")
                eeg_loger.is_recording = 1

                self.eeg_file_path = "test_eeg_data\\" + self.new_file_entry.get_text() + ".csv"
                eeg_loger.start_recording(self.eeg_file_path)

                self.import_button.set_sensitive(False)
                self.emotion_detect_button.set_sensitive(False)

    def on_file_choose(self, widget):
        """ Open the file chooser.

        Chosen file needs to be in exact format so only those files recorded be Emotiv SDK could be imported

        """
        dialog = Gtk.FileChooserDialog("Select EEG file", self, Gtk.FileChooserAction.OPEN,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

        response = dialog.run()

        eeg_file_path = ""
        if response == Gtk.ResponseType.OK:
            eeg_file_path = dialog.get_filename()
            self.imported_filename_label.set_label(basename(eeg_file_path))

        dialog.destroy()
        if eeg_file_path != "":

            self.eeg_file_path = eeg_file_path
            self.update_eeg_plot()

            # GObject.idle_add(self.update_eeg_plot)

    def update_eeg_plot(self):
        """ Update graph with eeg signal.

        At the end of recording or after file import graph should updated with this method.

        Parameters
        ----------
        eeg_file_path: string, path to the eeg file

        """
        self.clear_eeg_plot()

        eeg_singal = pd.read_csv(self.eeg_file_path)

        signal_f3 = eeg_singal[" 'F3'"]
        signal_f4 = eeg_singal[" 'F4'"]
        signal_af3 = eeg_singal[" 'AF3'"]
        signal_af4 = eeg_singal[" 'AF4'"]

        signal_f3 = self.normalize_signal(signal_f3)
        signal_f4 = self.normalize_signal(signal_f4)
        signal_af3 = self.normalize_signal(signal_af3)
        signal_af4 = self.normalize_signal(signal_af4)

        t = arange(0.0, len(signal_f3), 1)

        self.ax1.plot(t, signal_f3, color=self.plot_color)
        self.ax1.set_ylabel('F3', **self.yprops)

        self.ax2.plot(t, signal_f4, color=self.plot_color)
        self.ax2.set_ylabel('F4', **self.yprops)

        self.ax3.plot(t, signal_af3, color=self.plot_color)
        self.ax3.set_ylabel('AF3', **self.yprops)

        self.ax4.plot(t, signal_af4, color=self.plot_color)
        self.ax4.set_ylabel('AF4', **self.yprops)

    def on_detect_emotion(self, widget):
        """ Predict emotion from EEG signal.

        Display valence, arousal valence and change the picture of emotion.

        Notes:
        ----------
        EEG file need to be selected or recording has to be done.
        """
        if self.eeg_file_path != "":
            arousal, valence, detected_emotion = emotion_detection.predict_emotion(self.eeg_file_path)
            self.arousal_value_label.set_label(str("%.2f" % arousal[0]))
            self.valence_value_label.set_label(str("%.2f" % valence[0]))

            self.image.set_from_file(self.label_emotion(detected_emotion[0]))
        else:
            dialog = DialogExample(self, "Error", "You did not choose any EEG file or recorded any EEG signal!")
            dialog.run()
            dialog.destroy()
            print("")

    def label_emotion(self, x):
        return {
            0: happy_image_path,
            1: sad_image_path,
            2: disgusted_image_path,
            3: angry_image_path,
            4: scared_image_path,
            5: surprised_image_path,
            6: neutral_image_path
        }[x]


class DialogExample(Gtk.Dialog):

    def __init__(self, parent, title, text):
        Gtk.Dialog.__init__(self, title, parent, 0,
            (Gtk.STOCK_OK, Gtk.ResponseType.OK))

        self.set_default_size(150, 100)

        label = Gtk.Label(text)

        box = self.get_content_area()
        box.add(label)
        self.show_all()

win = GridWindow()
win.resize(1000, 300)
win.connect("delete-event", Gtk.main_quit)
win.show_all()
Gtk.main()
