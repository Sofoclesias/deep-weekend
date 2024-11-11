"""
Fuente original: https://github.com/Natooz/MidiTok/blob/main/miditok/classes.py
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence, Mapping
from math import log2
import warnings
from dataclasses import dataclass, field, replace


from .constants import (
    AC_NOTE_DENSITY_BAR,
    AC_NOTE_DENSITY_BAR_MAX,
    AC_NOTE_DENSITY_TRACK,
    AC_NOTE_DENSITY_TRACK_MAX,
    AC_NOTE_DENSITY_TRACK_MIN,
    AC_NOTE_DURATION_BAR,
    AC_NOTE_DURATION_TRACK,
    AC_PITCH_CLASS_BAR,
    AC_POLYPHONY_BAR,
    AC_POLYPHONY_MAX,
    AC_POLYPHONY_MIN,
    AC_POLYPHONY_TRACK,
    AC_REPETITION_TRACK,
    AC_REPETITION_TRACK_NUM_BINS,
    AC_REPETITION_TRACK_NUM_CONSEC_BARS,
    BEAT_RES,
    BEAT_RES_REST,
    CHORD_MAPS,
    CHORD_TOKENS_WITH_ROOT_NOTE,
    CHORD_UNKNOWN,
    DEFAULT_NOTE_DURATION,
    DELETE_EQUAL_SUCCESSIVE_TEMPO_CHANGES,
    DELETE_EQUAL_SUCCESSIVE_TIME_SIG_CHANGES,
    DRUM_PITCH_RANGE,
    ENCODE_IDS_SPLIT,
    LOG_TEMPOS,
    MANDATORY_SPECIAL_TOKENS,
    MAX_PITCH_INTERVAL,
    NUM_TEMPOS,
    NUM_VELOCITIES,
    ONE_TOKEN_STREAM_FOR_PROGRAMS,
    PITCH_BEND_RANGE,
    PITCH_INTERVALS_MAX_TIME_DIST,
    PITCH_RANGE,
    PROGRAM_CHANGES,
    PROGRAMS,
    REMOVE_DUPLICATED_NOTES,
    SPECIAL_TOKENS,
    SUSTAIN_PEDAL_DURATION,
    TEMPO_RANGE,
    TIME_SIGNATURE_RANGE,
    USE_CHORDS,
    USE_NOTE_DURATION_PROGRAMS,
    USE_PITCH_BENDS,
    USE_PITCH_INTERVALS,
    USE_PITCHDRUM_TOKENS,
    USE_PROGRAMS,
    USE_RESTS,
    USE_SUSTAIN_PEDALS,
    USE_TEMPOS,
    USE_TIME_SIGNATURE,
    USE_VELOCITIES,
)

def _format_special_token(token: str) -> str:
    parts = token.split("_")
    if len(parts) == 1:
        parts.append("None")
    elif len(parts) > 2:
        parts = ["-".join(parts[:-1]), parts[-1]]
        warnings.warn(
            f"miditok.TokenizerConfig: special token {token} must"
            " contain one underscore (_).This token will be saved as"
            f" {'_'.join(parts)}.",
            stacklevel=2,
        )
    return "_".join(parts)

class TokenizerConfig:
    def __init__(
        self,
        pitch_range: tuple[int, int] = PITCH_RANGE,
        beat_res: dict[tuple[int, int], int] = BEAT_RES,
        num_velocities: int = NUM_VELOCITIES,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
        encode_ids_split: Literal["bar", "beat", "no"] = ENCODE_IDS_SPLIT,
        use_velocities: bool = USE_VELOCITIES,
        use_note_duration_programs: Sequence[int] = USE_NOTE_DURATION_PROGRAMS,
        use_chords: bool = USE_CHORDS,
        use_rests: bool = USE_RESTS,
        use_tempos: bool = USE_TEMPOS,
        use_time_signatures: bool = USE_TIME_SIGNATURE,
        use_sustain_pedals: bool = USE_SUSTAIN_PEDALS,
        use_pitch_bends: bool = USE_PITCH_BENDS,
        use_programs: bool = USE_PROGRAMS,
        use_pitch_intervals: bool = USE_PITCH_INTERVALS,
        use_pitchdrum_tokens: bool = USE_PITCHDRUM_TOKENS,
        default_note_duration: int | float = DEFAULT_NOTE_DURATION,
        beat_res_rest: dict[tuple[int, int], int] = BEAT_RES_REST,
        chord_maps: dict[str, tuple] = CHORD_MAPS,
        chord_tokens_with_root_note: bool = CHORD_TOKENS_WITH_ROOT_NOTE,
        chord_unknown: tuple[int, int] = CHORD_UNKNOWN,
        num_tempos: int = NUM_TEMPOS,
        tempo_range: tuple[int, int] = TEMPO_RANGE,
        log_tempos: bool = LOG_TEMPOS,
        remove_duplicated_notes: bool = REMOVE_DUPLICATED_NOTES,
        delete_equal_successive_tempo_changes: bool = (
            DELETE_EQUAL_SUCCESSIVE_TEMPO_CHANGES
        ),
        time_signature_range: Mapping[
            int, list[int] | tuple[int, int]
        ] = TIME_SIGNATURE_RANGE,
        sustain_pedal_duration: bool = SUSTAIN_PEDAL_DURATION,
        pitch_bend_range: tuple[int, int, int] = PITCH_BEND_RANGE,
        delete_equal_successive_time_sig_changes: bool = (
            DELETE_EQUAL_SUCCESSIVE_TIME_SIG_CHANGES
        ),
        programs: Sequence[int] = PROGRAMS,
        one_token_stream_for_programs: bool = ONE_TOKEN_STREAM_FOR_PROGRAMS,
        program_changes: bool = PROGRAM_CHANGES,
        max_pitch_interval: int = MAX_PITCH_INTERVAL,
        pitch_intervals_max_time_dist: bool = PITCH_INTERVALS_MAX_TIME_DIST,
        drums_pitch_range: tuple[int, int] = DRUM_PITCH_RANGE,
        ac_polyphony_track: bool = AC_POLYPHONY_TRACK,
        ac_polyphony_bar: bool = AC_POLYPHONY_BAR,
        ac_polyphony_min: int = AC_POLYPHONY_MIN,
        ac_polyphony_max: int = AC_POLYPHONY_MAX,
        ac_pitch_class_bar: bool = AC_PITCH_CLASS_BAR,
        ac_note_density_track: bool = AC_NOTE_DENSITY_TRACK,
        ac_note_density_track_min: int = AC_NOTE_DENSITY_TRACK_MIN,
        ac_note_density_track_max: int = AC_NOTE_DENSITY_TRACK_MAX,
        ac_note_density_bar: bool = AC_NOTE_DENSITY_BAR,
        ac_note_density_bar_max: int = AC_NOTE_DENSITY_BAR_MAX,
        ac_note_duration_bar: bool = AC_NOTE_DURATION_BAR,
        ac_note_duration_track: bool = AC_NOTE_DURATION_TRACK,
        ac_repetition_track: bool = AC_REPETITION_TRACK,
        ac_repetition_track_num_bins: int = AC_REPETITION_TRACK_NUM_BINS,
        ac_repetition_track_num_consec_bars: int = AC_REPETITION_TRACK_NUM_CONSEC_BARS,
        **kwargs,
    ) -> None:
        # Checks
        if not 0 <= pitch_range[0] < pitch_range[1] <= 127:
            msg = (
                "`pitch_range` must be within 0 and 127, and an first value "
                f"greater than the second (received {pitch_range})"
            )
            raise ValueError(msg)
        if not 1 <= num_velocities <= 127:
            msg = (
                "`num_velocities` must be within 1 and 127 (received "
                f"{num_velocities})"
            )
            raise ValueError(msg)
        if max_pitch_interval and not 0 <= max_pitch_interval <= 127:
            msg = (
                "`max_pitch_interval` must be within 0 and 127 (received "
                f"{max_pitch_interval})."
            )
            raise ValueError(msg)
        if use_time_signatures:
            for denominator in time_signature_range:
                if not log2(denominator).is_integer():
                    msg = (
                        "`time_signature_range` contains an invalid time signature "
                        "denominator. MidiTok only supports powers of 2 denominators, "
                        f"does the MIDI protocol. Received {denominator}."
                    )
                    raise ValueError(msg)

        # Global parameters
        self.pitch_range: tuple[int, int] = pitch_range
        self.beat_res: dict[tuple[int, int], int] = beat_res
        self.num_velocities: int = num_velocities
        self.remove_duplicated_notes = remove_duplicated_notes
        self.encode_ids_split = encode_ids_split

        # Special tokens
        self.special_tokens: list[str] = []
        for special_token in list(special_tokens):
            token = _format_special_token(special_token)
            if token not in self.special_tokens:
                self.special_tokens.append(token)
            else:
                warnings.warn(
                    f"The special token {token} is present twice in your configuration."
                    f" Skipping its duplicated occurrence.",
                    stacklevel=2,
                )
        # Mandatory special tokens, no warning here
        for special_token in MANDATORY_SPECIAL_TOKENS:
            token = _format_special_token(special_token)
            if token not in self.special_tokens:
                self.special_tokens.append(token)

        # Additional token types params, enabling additional token types
        self.use_velocities: bool = use_velocities
        self.use_note_duration_programs: set[int] = set(use_note_duration_programs)
        self.use_chords: bool = use_chords
        self.use_rests: bool = use_rests
        self.use_tempos: bool = use_tempos
        self.use_time_signatures: bool = use_time_signatures
        self.use_sustain_pedals: bool = use_sustain_pedals
        self.use_pitch_bends: bool = use_pitch_bends
        self.use_programs: bool = use_programs
        self.use_pitch_intervals: bool = use_pitch_intervals
        self.use_pitchdrum_tokens: bool = use_pitchdrum_tokens

        # Duration
        self.default_note_duration = default_note_duration

        # Programs
        self.programs: set[int] = set(programs)
        # These needs to be set to False if the tokenizer is not using programs
        self.one_token_stream_for_programs = (
            one_token_stream_for_programs and use_programs
        )
        self.program_changes = program_changes and use_programs

        # Check for rest compatibility with duration tokens
        if self.use_rests and len(self.use_note_duration_programs) < 129:
            msg = (
                "Disabling rests tokens. `Rest` tokens are compatible when note "
                "`Duration` tokens are enabled."
            )
            if not self.use_programs:
                self.use_rests = False
                warnings.warn(
                    msg + " Your configuration explicitly disable `Program` (allowing"
                    "to tokenize any track) while disabling note `Duration` "
                    "tokens for some programs.",
                    stacklevel=2,
                )

            elif any(p not in self.use_note_duration_programs for p in self.programs):
                self.use_rests = False
                warnings.warn(
                    msg + "You enabled `Program` tokens while disabling note duration "
                    " tokens for programs (`use_note_duration_programs`) outside "
                    "of the supported `programs`.",
                    stacklevel=2,
                )

        # Rest params
        self.beat_res_rest: dict[tuple[int, int], int] = beat_res_rest
        if self.use_rests:
            max_rest_res = max(self.beat_res_rest.values())
            max_global_res = max(self.beat_res.values())
            if max_rest_res > max_global_res:
                msg = (
                    "The maximum resolution of the rests must be inferior or equal to"
                    "the maximum resolution of the global beat resolution"
                    f"(``config.beat_res``). Expected <= {max_global_res},"
                    f"{max_rest_res} was given."
                )
                raise ValueError(msg)

        # Chord params
        self.chord_maps: dict[str, tuple] = chord_maps
        # Tokens will look as "Chord_C:maj"
        self.chord_tokens_with_root_note: bool = chord_tokens_with_root_note
        # (3, 6) for chords between 3 and 5 notes
        self.chord_unknown: tuple[int, int] = chord_unknown

        # Tempo params
        self.num_tempos: int = num_tempos
        self.tempo_range: tuple[int, int] = tempo_range  # (min_tempo, max_tempo)
        self.log_tempos: bool = log_tempos
        self.delete_equal_successive_tempo_changes = (
            delete_equal_successive_tempo_changes
        )

        # Time signature params
        self.time_signature_range = {
            denominator: (
                list(range(numerators[0], numerators[1] + 1))
                if isinstance(numerators, tuple)
                else numerators
            )
            for denominator, numerators in time_signature_range.items()
        }
        self.delete_equal_successive_time_sig_changes = (
            delete_equal_successive_time_sig_changes
        )

        # Sustain pedal params
        self.sustain_pedal_duration = sustain_pedal_duration and self.use_sustain_pedals

        # Pitch bend params
        self.pitch_bend_range = pitch_bend_range

        # Pitch as interval tokens
        self.max_pitch_interval = max_pitch_interval
        self.pitch_intervals_max_time_dist = pitch_intervals_max_time_dist

        # Drums
        self.drums_pitch_range = drums_pitch_range

        # Pop legacy kwargs
        legacy_args = (
            ("nb_velocities", "num_velocities"),
            ("nb_tempos", "num_tempos"),
        )
        for legacy_arg, new_arg in legacy_args:
            if legacy_arg in kwargs:
                setattr(self, new_arg, kwargs.pop(legacy_arg))
                warnings.warn(
                    f"Argument {legacy_arg} has been renamed {new_arg}, you should"
                    " consider to updateyour code with this new argument name.",
                    stacklevel=2,
                )

        # Attribute controls
        self.ac_polyphony_track = ac_polyphony_track
        self.ac_polyphony_bar = ac_polyphony_bar
        self.ac_polyphony_min = ac_polyphony_min
        self.ac_polyphony_max = ac_polyphony_max
        self.ac_pitch_class_bar = ac_pitch_class_bar
        self.ac_note_density_track = ac_note_density_track
        self.ac_note_density_track_min = ac_note_density_track_min
        self.ac_note_density_track_max = ac_note_density_track_max
        self.ac_note_density_bar = ac_note_density_bar
        self.ac_note_density_bar_max = ac_note_density_bar_max
        self.ac_note_duration_bar = ac_note_duration_bar
        self.ac_note_duration_track = ac_note_duration_track
        self.ac_repetition_track = ac_repetition_track
        self.ac_repetition_track_num_bins = ac_repetition_track_num_bins
        self.ac_repetition_track_num_consec_bars = ac_repetition_track_num_consec_bars

        # Additional params
        self.additional_params = kwargs

    @property
    def max_num_pos_per_beat(self) -> int:
        """
        Returns the maximum number of positions per ticks covered by the config.

        :return: maximum number of positions per ticks covered by the config.
        """
        return max(self.beat_res.values())

    @property
    def using_note_duration_tokens(self) -> bool:
        """
        Return whether the configuration allows to use note duration tokens.

        :return: whether the configuration allows to use note duration tokens for at
            least one program.
        """
        return len(self.use_note_duration_programs) > 0
    
@dataclass
class Event:
    type_: str
    value: str | int
    time: int = -1
    program: int = 0
    desc: str | int = 0

    def __str__(self) -> str:
        """
        Return the string value of the ``Event``.

        :return: string value of the ``Event`` as a combination of its type and value.
        """
        return f"{self.type_}_{self.value}"

    def __repr__(self) -> str:
        """
        Return the representation of this ``Event``.

        :return: representation of the event.
        """
        return (
            f"Event(type={self.type_}, value={self.value}, time={self.time},"
            f" desc={self.desc})"
        )
    