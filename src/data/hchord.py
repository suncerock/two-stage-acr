import numpy as np
import re
from collections import OrderedDict
import music21 as m21
import warnings
warnings.filterwarnings("ignore")


""" Harte Chord Class -
    after Symbolic Representation of Musical Chords: a proposed syntax for text annotations 2005 ISMIR

"""

__author__ = 'Michael Kohl / Yiwei Ding'
__copyright__ = 'Copyright 2020, Audiolabs Erlangen, 2025 University of Würzburg'

__license__ = "GPL"
__version__ = "2.0.1"
__maintainer__ = "Yiwei Ding / Christof Weiß / Leo Brütting"
__email__ = 'michael_kohl@me.com, yiwei.ding@uni-wuerzburg.de'
__status__ = "Production"

_shorthands = OrderedDict([
    ('13', ['1', '3', '5', 'b7', '9', '11', '13']),
    ('maj13', ['1', '3', '5', '7', '9', '11', '13']),
    ('min13', ['1', 'b3', '5', 'b7', '9', '11', '13']),
    ('11', ['1', '3', '5', 'b7', '9', '11']),
    ('min11', ['1', 'b3', '5', 'b7', '9', '11']),
    ('9', ['1', '3', '5', 'b7', '9']),
    ('maj9', ['1', '3', '5', '7', '9']),
    ('min9', ['1', 'b3', '5', 'b7', '9']),
    ('7', ['1', '3', '5', 'b7']),
    ('maj7', ['1', '3', '5', '7']),
    ('min7', ['1', 'b3', '5', 'b7']),
    ('dim7', ['1', 'b3', 'b5', 'bb7']),
    ('hdim7', ['1', 'b3', 'b5', 'b7']),
    ('minmaj7', ['1', 'b3', '5', '7']),
    ('aug7', ['1', '3', '#5', 'b7']),
    ('7sus4', ['1', '4', '5', 'b7']),
    ('maj6', ['1', '3', '5', '6']),
    ('min6', ['1', 'b3', '5', '6']),
    ('maj', ['1', '3', '5']),
    ('min', ['1', 'b3', '5']),
    ('dim', ['1', 'b3', 'b5']),
    ('aug', ['1', '3', '#5']),
    ('sus4', ['1', '4', '5']),
    ('sus2', ['1', '2', '5']),
    ('5', ['1', '5']),
    ('1', ['1'])
])


SHORTHAND_ALIASES = {'6': 'maj6', '7sus': '7sus4'}


# --- Chord Label Parsing ---
# This monster regexp is pulled from the JAMS chord namespace,
# which is in turn derived from the context-free grammar of
# Harte et al., 2005.
CHORD_RE = re.compile(
    r"^((N|X)|(([A-G](b*|#*))((:(maj|min|dim|aug|1|5|sus2|sus4|maj6|min6|7|maj7|min7|dim7|hdim7|minmaj7|aug7|7sus4|9|maj9|min9|11|maj11|min11|13|maj13|min13|6|7sus)(\((\*?((b*|#*)([1-9]|1[0-3]?))(,\*?((b*|#*)([1-9]|1[0-3]?)))*)\))?)|(:\((\*?((b*|#*)([1-9]|1[0-3]?))(,\*?((b*|#*)([1-9]|1[0-3]?)))*)\)))?((/((b*|#*)([1-9]|1[0-3]?)))?)?))$"
)  # nopep8

class Hchord:
    def __init__(self, input_string: str = None, root=None, shorthand=None, intervals=None, bass=None):
        self._intervals: set[str] = set()

        if input_string is not None:
            input_string = input_string.replace(' ', '')
            if root is not None or shorthand is not None or intervals is not None or bass is not None:
                raise ValueError('String input and keyword input not possible simultaneously')
            
            # if not CHORD_RE.match(input_string):
            #     warnings.warn("Invalid chord label: " "{}".format(input_string))

            
            qua = None

            if input_string == 'N':  # 'N' (No) chord
                root = None
                bass = None
            elif input_string.endswith('None'):
                root = None
                bass = None
                warnings.warn(f'Chord {input_string} ends with None is non-standard Harte chord!')
            elif input_string == 'X':  # 'X' (Unknown) chord
                root = None
                bass = None
                warnings.warn('Chord X (unknown) is non-standard Harte chord!')
            else:  # Start parsing the input string
                if '/' in input_string:  # Inversion
                    root_qua, bass = input_string.split('/')
                else: # No inversion
                    root_qua = input_string
                    bass = '1'

                if not ':' in root_qua:  # No quality given, C = C:maj
                    root = root_qua
                    qua = 'maj'
                else:  # Quality given
                    root, qua = root_qua.split(':')
                    # TODO: Fix these weird chords
                    if qua == '(*5)':  # Special case: C:(*5) = C:maj(*5)
                        qua = 'maj(*5)'
                    elif qua == 'maj7(*b5)':  # What is b5 in maj7???
                        qua = 'maj7'

                if not '(' in qua:  # No additional intervals given
                    shorthand = qua
                else:
                    shorthand, intervals = qua.split('(')
                    if shorthand == '':  # Intervals without shorthand
                        shorthand = None
                    intervals = intervals.rstrip(')')
        else:  # root, shorthand, intervals, bass input
            if bass is None:
                bass = '1'  # default bass

        # add root
        self.root = root

        # add shorthand
        if shorthand is not None:
            shorthand = SHORTHAND_ALIASES.get(shorthand, shorthand)  # non-standard alias replacement
            if shorthand in _shorthands:
                self._add_shorthand(shorthand)
            else:
                warnings.warn(f'shorthand {shorthand} not valid!')
                self.root = None  # invalidate chord
                intervals = None  # do not add intervals
                bass = None  # do not add bass

        # add intervals
        if intervals is not None:
            self.add_interval(intervals.split(','))

        # add bass interval
        self.bass_interval = bass

    def __eq__(self, value):
        if not isinstance(value, Hchord):
            return False
        return (
            self.root == value.root and
            self.intervals == value.intervals and
            self.bass_interval == value.bass_interval
        )

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, new_root: str = None):
        if new_root is None:
            self._root = None
        elif self._is_note(new_root):
            self._root = new_root
        else:
            raise TypeError('root has to be a valid combination out of A, B, C, D, E, F, G and #,b')

    @property
    def intervals(self):
        return frozenset(self._intervals)

    @property
    def bass_interval(self):
        return self._bass_interval

    @bass_interval.setter
    def bass_interval(self, new_bass: str = None):
        if new_bass is None:  # only for N and X
            self._bass_interval = None
            return

        if self._is_interval(new_bass):
            self._bass_interval = new_bass
            if new_bass not in self._intervals:
                warnings.warn('bass interval not in chord intervals! Added bass interval ' + new_bass + ', while chord intervals are ' + str(self._intervals))
                
        elif self._is_note(new_bass):
            m21_bass = m21.pitch.Pitch(new_bass.replace('b', '-'))
            m21_root = m21.pitch.Pitch(self.root.replace('b', '-'))
            m21_int = m21.interval.Interval(noteStart=m21_root, noteEnd=m21_bass)
            if '-' in m21_int.directedName:
                m21_int = m21_int.complement

            new_bass = self.m21_interval_to_interval(m21_int)
            # FIXME: temporary fix for some weird cases in SWD
            if new_bass == 'b1':
                new_bass = '7'
            if new_bass == '#2' and 'b3' in self._intervals:
                new_bass = 'b3'
            if new_bass == 'b4' and '3' in self._intervals:
                new_bass = '3'
            if new_bass == 'bb6' and '5' in self._intervals:
                new_bass = '5'

            self.bass_interval = new_bass
        else:
            raise TypeError(f'Bass interval {new_bass} is neither a valid interval nor a note!')

    def add_interval(self, new_interval: str|list):
        if isinstance(new_interval, list):
            if '1' not in new_interval: # FIXME: some datasets do not encode (1)
                new_interval.append('1')
            for val in new_interval:
                self.add_interval(val)
            return

        if new_interval.startswith('*'):
            self.remove_interval(new_interval.lstrip('*'))
        elif self._is_interval(new_interval):
            # check if generic interval already existent
            for val in self._intervals:
                if (new_interval.lstrip('b#') == val.lstrip('b#')) and (new_interval != val):
                    err = 'base interval already used! Added ' + new_interval + ', while ' + val + ' already existent.'
                    warnings.warn(err)
            self._intervals.add(new_interval)
        else:
            raise TypeError(f'interval {new_interval} is not a valid interval!')

    def remove_interval(self, del_interval):
        if self._is_interval(del_interval):
            if del_interval == '3' and 'b3' in self._intervals:  # FIXME: removing ambiguous intervals in idols song jp
                assert '3' not in self._intervals, '3 and b3 could not be both in the chord'
                self._intervals.remove('b3')
            elif del_interval in self._intervals:
                self._intervals.remove(del_interval)
            else:
                warnings.warn(f'Removing an interval {del_interval} not in the chord!')
        else:
            raise TypeError(f'interval {del_interval} is not a valid interval!')

    def _add_shorthand(self, shorthand):
        self.add_interval(_shorthands[shorthand])

    def __repr__(self):
        return self.export_string()

    def export_string(self, style='extended', absolute_bass=False):
        if self.root is None:
            return 'N'

        bass_string = ''
        if self.bass_interval != '1':
            if absolute_bass:
                m21_root = m21.pitch.Pitch(self.root.replace('b', '-'))
                m21_int =self.interval_to_m21_interval(self.bass_interval)
                m21_bass = m21_int.transposePitch(m21_root)
                bass_string = '/' + str(m21_bass).replace('-', 'b')
            else:
                bass_string = '/' + self.bass_interval

        if style == 'shorthand':
            best_shorthand, best_interval_list = None, None
            for shorthand, intervals in _shorthands.items():
                if set(intervals).issubset(self._intervals):
                    best_shorthand = shorthand
                    best_interval_list = list(self._intervals.difference(set(intervals)))
                    break
            
            if best_shorthand is None:
                return self.export_string(style='extended', absolute_bass=absolute_bass)
            if best_shorthand != '1':  # Because 1 often matches and is not a valid shorthand
                separator = ','
                if len(best_interval_list) > 0:
                    interval_string = separator.join(sorted(best_interval_list, key=lambda x:x.lstrip('b#')))
                    return f'{self.root}:{best_shorthand}({interval_string}){bass_string}'
                else:
                    return f'{self.root}:{best_shorthand}{bass_string}'
            else:
                if self.intervals == {'1'}:
                    return f'{self.root}:1{bass_string}'
                else:
                    return self.export_string(style='extended', absolute_bass=absolute_bass)


        if style == 'extended':
            interval_list = list(self.intervals.difference({'1'}))
            if len(interval_list) == 0:
                return f'{self.root}:(1){bass_string}'
            separator = ','
            interval_string = separator.join(sorted(interval_list, key=lambda x:x.lstrip('b#')))
            return (self.root + ':(' + interval_string + ')' + bass_string)

        if style == 'majmin':
            warnings.warn('majmin style is deprecated. Use triad instead.', DeprecationWarning)
            return self.export_string(style='majminInv', absolute_bass=absolute_bass).split('/')[0]

        if style == 'majminInv':
            warnings.warn('majminInv style is deprecated. Use triadInv instead.', DeprecationWarning)
            if 'b3' in self._intervals and '3' not in self._intervals:
                return f'{self.root}:min{bass_string}'
            elif '3' in self._intervals and 'b3' not in self._intervals:
                return f'{self.root}:maj{bass_string}'
            else:
                return 'N'

        if style == 'triad':
            return self.export_string(style='triadInv', absolute_bass=absolute_bass).split('/')[0]

        if style == 'triadInv':
            if 'b3' in self._intervals and '3' not in self._intervals:
                qua = 'dim' if 'b5' in self._intervals else 'min'
                return f'{self.root}:{qua}{bass_string}'
            elif '3' in self._intervals and 'b3' not in self._intervals:
                qua = 'aug' if '#5' in self._intervals else 'maj'
                return f'{self.root}:{qua}{bass_string}'
            else:
                return 'N'

        # wrong style
        raise ValueError(f'style {style} not valid')
    
    def encode(self, absolute=False, enforce_root=False, enforce_bass=False, include_extended=False):
        if self.root is None:
            return -1, np.zeros(12, dtype=int), -1

        m21_root = m21.pitch.Pitch(self.root.replace('b', '-'))
        root_number = int(m21_root.pitchClass)

        bass_number = self.interval_to_halfsteps(self.bass_interval) % 12

        pitches_arr = np.zeros(12, dtype=int)

        intervals = self.intervals
        if not include_extended:
            intervals = {iv for iv in self.intervals if int(iv.lstrip('b#')) <= 7}
        for interval in intervals:
            pitches_arr[(self.interval_to_halfsteps(interval)) % 12] = 1
        
        if enforce_root:
            pitches_arr[0] = 1
        if enforce_bass:
            pitches_arr[bass_number] = 1

        if absolute:
            bass_number = (root_number + bass_number) % 12
            pitches_arr = np.roll(pitches_arr, root_number)

        return root_number, pitches_arr, bass_number

    def transpose(self, interval: str):
        downwards = False

        if interval.startswith('-'):
            interval=interval.lstrip('-')
            downwards = True

        if self._is_interval(interval):
            m21_root = m21.pitch.Pitch(self.root.replace('b', '-'))
            m21_int = self.interval_to_m21_interval(interval)
            if downwards:
                m21_int=m21_int.reverse()
            m21_root = m21_int.transposePitch(m21_root)
            self.root = m21_root.name.replace('-', 'b')
        else:
            raise TypeError(f'interval {interval} is not a valid interval!')

    def _is_note(self, candidate):
        return re.match(r'^[A-G](b*|#*)$', candidate) is not None

    def _is_interval(self, candidate):
        return re.match(r'^(b*|#*)([1-9]|1[0-3]?)$', candidate) is not None

    @staticmethod
    def interval_to_halfsteps(interval: str):
        init_halfsteps = [0, 2, 4, 5, 7, 9, 11, 0, 2, 4, 5, 7, 9]
        halfsteps = init_halfsteps[int(interval.lstrip('b#')) - 1]
        while interval.startswith('#') or interval.startswith('b'):
            if interval.startswith('#'):
                halfsteps += 1
                interval = interval[1:]
            if interval.startswith('b'):
                halfsteps -= 1
                interval = interval[1:]

        return halfsteps

    @staticmethod
    def interval_to_m21_interval(interval: str):
        degree = int(interval.lstrip('b#'))
        assert degree in [1,2,3,4,5,6,7], 'generic interval must be between 1 and 7'
        if degree in [1, 4, 5]:
            if (not interval.startswith('b')) and (not interval.startswith('#')):
                return m21.interval.Interval('P' + str(degree))
            else:
                interval = interval.replace('#', 'A')
                interval = interval.replace('b', 'd')
                return m21.interval.Interval(interval)
        else:
            if (not interval.startswith('b')) and (not interval.startswith('#')):
                return m21.interval.Interval('M' + str(degree))
            else:
                if not interval.startswith('b'):
                    interval = interval.replace('#', 'A')
                    return m21.interval.Interval(interval)
                else:
                    if interval[1] != 'b':
                        interval = interval.replace('b', 'm')
                        return m21.interval.Interval(interval)
                    else:
                        interval = interval[1:].replace('b', 'd')
                        return m21.interval.Interval(interval)

    @staticmethod
    def m21_interval_to_interval(m21_int: m21.interval.Interval):
        if '-' in m21_int.directedName:
            raise Exception('Negative intervals handled outside! How can this happen????')

        interval = m21_int.name
        degree = m21_int.generic.directed
        assert degree in [1,2,3,4,5,6,7], 'generic interval must be between 1 and 7'

        if degree in [1, 4, 5]: # perfect intervals
            if interval.startswith('P'):
                return interval.lstrip('P')
            else:
                interval = interval.replace('A', '#')
                interval = interval.replace('d', 'b')
                return interval
        else:
            if interval.startswith('M'):
                return interval.lstrip('M')
            elif interval.startswith('m'):
                return interval.replace('m', 'b')
            elif interval.startswith('d'):
                return 'b' * (interval.count('d') + 1) + interval.lstrip('d')
            else:
                return interval.replace('A', '#')
