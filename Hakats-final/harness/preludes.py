"""код перед каждым скриптом модели: данные уже загружены, и скрипты выходят короткими"""

import builtins
import keyword

# CSV лежат в T[путь] и в rows, если файл один
# несуществующая колонка кидает MissingColumn даже через .get, иначе модель подставит значение сама
CSV_PRELUDE = """import csv, glob, os, sys
class MissingColumn(KeyError):
    pass
class _Row(dict):
    def __missing__(self, key):
        raise MissingColumn(key)
    def get(self, key, default=None):
        return self[key]
    def __contains__(self, key):
        if not dict.__contains__(self, key):
            raise MissingColumn(key)
        return True
def _report_missing_column(kind, error, tb, _default=sys.excepthook):
    if issubclass(kind, MissingColumn):
        print('IMPOSSIBLE: the data has no column ' + repr(error.args[0]))
    else:
        _default(kind, error, tb)
sys.excepthook = _report_missing_column
def _number(value):
    for convert in (int, float):
        try:
            return convert(value)
        except (TypeError, ValueError):
            pass
    return (value or '').strip()
T = {}
for _path in sorted(glob.glob('**/*.csv', recursive=True)):
    with open(_path, newline='', encoding='utf-8-sig') as _handle:
        _first = _handle.readline()
        _handle.seek(0)
        T[_path.replace(os.sep, '/')] = [_Row({k: _number(v) for k, v in row.items()})
                                         for row in csv.DictReader(_handle, delimiter=max(',;\t|', key=_first.count))]
rows = next(iter(T.values())) if len(T) == 1 else None
"""

# записи markdown лежат в KB[имя], псевдонимы и имена файлов тоже находятся
# одно имя у двух записей кидает AmbiguousName
KB_PRELUDE = """import glob, os, sys
class AmbiguousName(KeyError):
    pass
def _report_ambiguous_name(kind, error, tb, _default=sys.excepthook):
    if issubclass(kind, AmbiguousName):
        print('IMPOSSIBLE: several records are named ' + repr(error.args[0]))
    else:
        _default(kind, error, tb)
sys.excepthook = _report_ambiguous_name
class _KnowledgeBase(dict):
    aliases = {}
    ambiguous = set()
    def _resolve(self, key):
        if dict.__contains__(self, key):
            return key
        if isinstance(key, str):
            for variant in (key, key.strip(), key.strip().lower()):
                if variant in self.ambiguous:
                    raise AmbiguousName(key)
                if variant in self.aliases:
                    return self.aliases[variant]
        return None
    def __missing__(self, key):
        title = self._resolve(key)
        if title is None:
            raise KeyError(key)
        return dict.__getitem__(self, title)
    def get(self, key, default=None):
        title = self._resolve(key)
        return default if title is None else dict.__getitem__(self, title)
    def __contains__(self, key):
        return self._resolve(key) is not None
KB = _KnowledgeBase()
_records = []
for _path in sorted(glob.glob('**/*.md', recursive=True)):
    _stem = os.path.splitext(os.path.basename(_path))[0]
    _record = {'_file': _path.replace(os.sep, '/'), '_kind': _stem.split('_')[0], '_aliases': []}
    for _line in open(_path, encoding='utf-8-sig'):
        if _line.startswith('# '):
            _record['_title'] = _line[2:].strip()
        elif ':' in _line:
            _field, _value = _line.split(':', 1)
            _value = _value.strip()
            if len(_value) >= 2 and _value[0] == _value[-1] and _value[0] in ('"', "'"):  # Employer: "Fjord Data"
                _value = _value[1:-1]
            _record[_field.strip()] = _value
            if 'known as' in _field.lower() or 'alias' in _field.lower():
                _record['_aliases'] += [a.strip() for a in _value.split(',') if a.strip()]
    _record.setdefault('_title', _record['_file'])
    _record.setdefault('Name', _record['_title'])
    _records.append((_stem, _record))
_titles = [r['_title'] for _, r in _records]
KB.ambiguous = {t for t in _titles if _titles.count(t) > 1}
KB.ambiguous |= {t.lower() for t in KB.ambiguous}
_other_names = []
for _stem, _record in _records:
    _key = _record['_file'] if _record['_title'] in KB.ambiguous else _record['_title']
    KB[_key] = _record
    _other_names += [(a, _key) for a in _record.pop('_aliases')]
    _other_names += [(_stem, _key), (_record['_file'], _key), (_record['_title'].lower(), _key)]
for _name, _key in _other_names:
    if not dict.__contains__(KB, _name) and _name not in KB.ambiguous:
        KB.aliases.setdefault(_name, _key)
        KB.aliases.setdefault(_name.lower(), _key)
"""

CSV_HINT = ("`T[path]` = list of row dicts for every CSV (numbers already int/float, blanks are ''), "
            "`rows` = the rows when there is a single CSV, e.g. "
            "`sum(r['units'] for r in rows if r['product'] == 'widget')`. A column that is not in the file "
            "raises an error (do not use .get defaults for it).")

KB_HINT = ("`KB[name]` = dict of fields for every markdown record, one entry per record; lookups by an "
           "alias (`Also known as`) work too. Keys are the `Field` names plus `_title`, `_file`, `_kind` = "
           "file prefix such as person/company; a missing field is absent or ''. E.g. "
           "`KB['Ann Lee']['Employer']` -> 'Fjord Data', `KB['Fjord Data']['Headquarters']` -> 'Oslo'. "
           "Follow exactly the chain of fields the question names.")

# таблицы markdown лежат в TABLES[имя файла], чтобы модель не разбирала их сама
MD_TABLES_PRELUDE = """import glob, os
def _cell(value):
    value = value.strip()
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            pass
    return value
TABLES = {}
for _path in sorted(glob.glob('**/*.md', recursive=True)):
    _rows, _header = [], None
    for _line in open(_path, encoding='utf-8-sig'):
        _line = _line.strip()
        if not (_line.startswith('|') and _line.endswith('|')):
            continue
        _cells = [c.strip() for c in _line.strip('|').split('|')]
        if all(set(c) <= set('-: ') for c in _cells):
            continue
        if _header is None:
            _header = _cells
        else:
            _rows.append({k: _cell(v) for k, v in zip(_header, _cells)})
    if _header is not None:
        TABLES[os.path.splitext(os.path.basename(_path))[0]] = _rows
"""

MD_TABLES_HINT = ("`TABLES[name]` = list of row dicts for every markdown table, keyed by the file name without "
                  ".md (e.g. TABLES['events']), numbers already int/float. Apply every rule from the rule "
                  "documents exactly as written.")

# записи ## ID лежат в RECORDS[файл][ID], нет поля значит MissingField даже через .get
SECTIONS_PRELUDE = """import glob, json, os, sys
class MissingField(KeyError):
    pass
class _Record(dict):
    # поля, которые есть хоть у одной записи этого файла: такое поле просто необязательное (у одних link, у других value)
    # невыполнимой задачу делает только поле, которого нет ни у одной записи
    _known = frozenset()
    def __missing__(self, key):
        if key in self._known:
            raise KeyError(key)
        raise MissingField(key)
    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in self._known:
            return default
        raise MissingField(key)
    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True
        if key in self._known:
            return False
        raise MissingField(key)
def _report_missing_field(kind, error, tb, _default=sys.excepthook):
    if issubclass(kind, MissingField):
        print('IMPOSSIBLE: a record has no field ' + repr(error.args[0]))
    else:
        _default(kind, error, tb)
sys.excepthook = _report_missing_field
def _value(text):
    text = text.strip()
    # досье бывают в виде yaml: status: "confirmed", rank: ["B46", "C46"], active: false
    # без разбора кавычки остаются в строке, список итерируется по буквам, а 'false' истинно, и всё молча неверно
    if text[:1] in ('"', '[', '{') or text in ('true', 'false', 'null'):
        try:
            return json.loads(text)
        except ValueError:
            pass
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]  # 'одинарные' json не читает
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text
RECORDS = {}
for _path in sorted(glob.glob('**/*.md', recursive=True)):
    _records, _current = {}, None
    for _line in open(_path, encoding='utf-8-sig'):
        if _line.startswith('## '):
            _current = _records.setdefault(_line[3:].strip(), _Record())
            _current.setdefault('id', _line[3:].strip())  # модель часто ищет ID записи среди полей
        elif _current is not None and ':' in _line and not _line.startswith('#'):
            _field, _text = _line.split(':', 1)
            _current[_field.strip()] = _value(_text)
    if _records:
        _known = frozenset(_field for _record in _records.values() for _field in _record)
        for _record in _records.values():
            _record._known = _known
        RECORDS[os.path.splitext(os.path.basename(_path))[0]] = _records
"""

# ID только строкой, голый id модель путает со встроенной функцией
SECTIONS_HINT = ("`RECORDS[name][record_id]` = dict of fields for every `## <record_id>` record in a markdown file, keyed "
                 "by the file name without .md (e.g. RECORDS['roster']['T1']['zone']), values already parsed (numbers, "
                 "true/false, quoted strings without quotes, [lists]); each dict also has key 'id'. Write record IDs "
                 "as string literals like 'T1' above (there is no variable `id`). "
                 "A field that no record of the file has raises an error that marks the task impossible; a field "
                 "only some records have is optional: test it with `in` or .get. Apply every rule from the rule documents exactly.")

# ходовые модули заранее, модель часто забывает импорт
BASE_PRELUDE = """import collections, functools, itertools, json, math, re
from fractions import Fraction
INF = float('inf')
"""

MAX_FILE_VARIABLES = 8  # если файлов с данными больше, переменные по именам не раздаём
# имена, которые уже заняты загрузчиками, под них файл не подставляем
_TAKEN_NAMES = {"T", "rows", "KB", "TABLES", "RECORDS", "INF", "Fraction", "collections", "functools", "itertools",
                "json", "math", "re", "csv", "glob", "os", "sys"}


def file_variables(files: list[str]) -> dict[str, str]:
    """какие CSV и JSON дать скрипту переменной по имени файла (kb/tree.json -> tree)"""
    names: dict[str, str] = {}
    data_files = [path for path in files if path.rpartition(".")[2] in ("csv", "json")]
    if len(data_files) > MAX_FILE_VARIABLES:
        return {}
    for path in data_files:
        stem = path.rpartition("/")[2].rpartition(".")[0]
        if not stem.isidentifier() or keyword.iskeyword(stem) or hasattr(builtins, stem) or stem in _TAKEN_NAMES:
            continue
        if stem in names:  # два файла с одним именем в разных папках: не угадываем, какой из них нужен
            names[stem] = ""
            continue
        names[stem] = path
    return {name: path for name, path in names.items() if path}


def file_variables_prelude(names: dict[str, str]) -> str:
    return f"""for _name, _path in {names!r}.items():
    if _path.endswith('.csv'):
        globals()[_name] = T[_path]
    else:
        try:
            with open(_path, encoding='utf-8-sig') as _handle:
                globals()[_name] = json.load(_handle)
        except Exception:
            pass
"""


def file_variables_hint(names: dict[str, str]) -> str:
    # перечисляем все имена, по одному примеру модель их не угадывает
    listed = ", ".join(f"{path} -> `{name}`" for name, path in names.items())
    return f"these files are loaded into variables: {listed}"
