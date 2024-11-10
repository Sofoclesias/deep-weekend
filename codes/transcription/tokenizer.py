import numpy as np
import tensorflow as tf
from .constants import *
import math



def tempo_qpm_to_mspq(tempo_qpm: int | float | np.ndarray):
        return 60000000 / tempo_qpm

class vocabulary:
    def __init__(self,tokenizer_config):
        self.config = tokenizer_config
        
        self.velocities = self.__create_velocities()
        self.time_signatures = self.__create_time_signatures()
        self.durations = self.__create_durations_tuples()
        self.rest = self.__create_rests()
        
        self.tempos = self.__create_tempos()
        self._tempos_mspq: np.ndarray = tempo_qpm_to_mspq(self.tempos)
        self._tempos_mspq.sort()
        self.default_tempo = self.tempos[np.argmin(np.abs(self.tempos - TEMPO))]

        
    def __create_velocities(self):
        return np.linspace(
                0, 127, self.config.num_velocities + 1, dtype=np.intc
            )[1:]
        
    def __create_time_signatures(self):
        time_signature_range = self.config.time_signature_range

        time_signatures = []
        for beat_res, beats in time_signature_range.items():
            if beat_res <= 0 or not math.log2(beat_res).is_integer():
                msg = (
                    f"The beat resolution ({beat_res}) in time signature must be a"
                    f"power of 2."
                )
                raise ValueError(msg)

            time_signatures.extend([(num_beats, beat_res) for num_beats in beats])

        return time_signatures

    def __create_durations_tuples(self):
        durations = []
        for beat_range, beat_res in self.config.beat_res.items():
            durations += [
                (beat, pos, beat_res)
                for beat in range(*beat_range)
                for pos in range(beat_res)
            ]
        durations += [
            (
                max(max(self.config.beat_res)),
                0,
                self.config.beat_res[max(self.config.beat_res)],
            )
        ]  # the last one
        del durations[0]  # removes duration of 0
        return durations
    
    def __create_rests(self):
        rests = []
        for beat_range, beat_res in self.config.beat_res_rest.items():
            rests += [
                (beat, pos, beat_res)
                for beat in range(*beat_range)
                for pos in range(beat_res)
            ]
        rests += [
            (
                max(max(self.config.beat_res_rest)),
                0,
                self.config.beat_res_rest[max(self.config.beat_res_rest)],
            )
        ]  # the last one
        del rests[0]  # removes rests of 0
        return rests
    
    def __create_tempos(self):
        tempo_fn = np.geomspace if self.config.log_tempos else np.linspace
        return tempo_fn(*self.config.tempo_range, self.config.num_tempos).round(2)

    def from_midi_tags(self):
        '''
        PAD -
        BOS -
        EOS -
        MASK -
        NoteOn -
        NoteOff -
        Velocity -
        Duration -
        Rest -
        TimeShift -
        Tempo -
        Program -
        TimeSig -
        '''
        vocab = []
        vocab += [f'{token}_{None}' for token in SPECIAL_TOKENS]
        vocab += [f'NoteOn_{i}' for i in range(*self.config.pitch_range)]
        vocab += [f'NoteOff_{i}' for i in range(*self.config.pitch_range)]
        vocab += [f"Velocity_{i}" for i in self.velocities]
        vocab += [f"Duration_{'.'.join(map(str, duration))}"for duration in self.durations]
        vocab += [f'Rest_{".".join(map(str, rest))}' for rest in self.rest]
        vocab += [f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures]
        vocab += [f"Tempo_{i}" for i in self.tempos]
        vocab += [f'TimeShift_{".".join(map(str, duration))}' for duration in self.durations]
        
        return vocab