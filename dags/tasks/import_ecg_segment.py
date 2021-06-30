"""import_ecg_Segment script

This script creates and exports a DataFrame including an ECG signal. It takes
into consideration several elements to load corresponding EDF file from the
server, such as the channel, start and end date.

This file can also be imported as a module and contains the following
class:

    * EdfLoader - A class used to load an edf file and return it in
    DataFrame format

and following function:
    * main - the main function of the script

"""

import pandas as pd
from pyedflib import highlevel
from pyedflib import edfreader
import argparse


class EdfLoader:
    """A class used to load an edf file and return it in DataFrame format

    Attributes
    ----------
    patient : str
        Patient to load
    record : str
        Record to load
    segment : str
        Segment to load
    edf_file_path : str
        Path of the EDF file to load
    headers : dict
        Headers of the EDF file
    channels : list
        Channels availiable if EDF file
    startdate : pd.DateTime
        Date of the beginning of the record
    sampling_frequency_hz: int
        Frequency of the sample in hz

    Methods
    -------
    convert_edf_to_dataframe(channel_name, start_time, end_time)
        Load EDF file and wrangle it into the DataFrame format

    """

    def __init__(self,
                 patient: str,
                 record: str,
                 segment: str):
        """__init__

        Parameters
        ----------
        patient : str
            Patient to load
        record : str
            Record to load
        segment : str
            Segment to load

        """
        default_path = '/home/DATA/lateppe/Recherche_ECG/'
        self.patient = patient
        self.record = record
        self.segment = segment
        self.edf_file_path = (f'{default_path}{self.patient}/EEG_'
                              f'{self.record}_{self.segment}.edf')

        self.headers = highlevel.read_edf_header(self.edf_file_path)
        self.channels = self.headers['channels']
        self.startdate = pd.to_datetime(
            self.headers['startdate']) + pd.Timedelta(hours=1)

    def convert_edf_to_dataframe(self,
                                 channel_name: str,
                                 start_time: pd.Timestamp,
                                 end_time: pd.Timestamp) -> pd.DataFrame:

        """Extract the ECG signal for a channel and export it to DataFrame
        format, limited by requested start time and end time

        Parameters
        ----------
        channel_name : str
            Name of the channel to load
        start_date : pd.Timestamp
            Start of the ECG signal to filter
        end_date : pd.Timestamp
            Start of the ECG signal to filter

        Returns
        -------
        df_ecg : pd.DataFrame
            DataFrame of the ECG for the requested channel, filter by
            with start and end timestamps

        """
        self.sampling_frequency_hz = int(self.headers[
            'SignalHeaders'][
            self.channels.index(channel_name)]['sample_rate'])

        with edfreader.EdfReader(self.edf_file_path) as f:
            signals = f.readSignal(self.channels.index(channel_name))

        freq_ns = 1_000_000_000 / self.sampling_frequency_hz
        df_ecg = pd.DataFrame(signals,
                              columns=['signal'],
                              index=pd.date_range(self.startdate,
                                                  periods=len(signals),
                                                  freq=f'{freq_ns}ns'
                                                  ))

        df_ecg = df_ecg[(df_ecg.index >= start_time) &
                        (df_ecg.index < end_time)]

        return df_ecg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument('-p',
                        '--patient',
                        dest='patient',
                        help='patient to load',
                        metavar='FILE')
    parser.add_argument('-r',
                        '--record',
                        dest='record',
                        help='record to load',
                        metavar='FILE')
    parser.add_argument('-ch',
                        '--channel',
                        dest='channel',
                        help='channel to load',
                        metavar='FILE')
    parser.add_argument('-sg',
                        '--segment',
                        dest='segment',
                        help='segment to load',
                        metavar='FILE')
    parser.add_argument('-st',
                        '--start_time',
                        dest='start_time',
                        help='start time for filter',
                        metavar='FILE')
    parser.add_argument('-et',
                        '--end_time',
                        dest='end_time',
                        help='end time for filter',
                        metavar='FILE')
    parser.add_argument('-ids',
                        '--annot_ids',
                        dest='annot_ids',
                        help='ids of annotators',
                        metavar='FILE')
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        help='output_folder_for_df',
                        metavar='FILE',
                        default='./exports')
    args = parser.parse_args()

    loader = EdfLoader(patient=args.patient,
                       record=args.record,
                       segment=args.segment)

    df_ecg = loader.convert_edf_to_dataframe(
        channel_name=args.channel,
        start_time=pd.Timestamp(args.start_time),
        end_time=pd.Timestamp(args.end_time))

    df_ecg.to_csv(f'{args.output_folder }/'
                  f'ecg_segment_'
                  f'{args.patient}_{args.record}_{args.channel}.csv',
                  index=False)
