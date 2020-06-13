def process_flags(all_flags):
    """
    Argument:
    Return:
        list of
    """
    fixed_map = {'fixed': True, 'param': False}

    flags = [Flags(name, fixed_map[flag_type]) for name, flag_type in all_flags.items()]

    return flags


def process_hparams(all_hparams):
    """
    Argument:
    Return:
    """
    hparams = [Hparams(name, values) for name, values in all_hparams.items()]

    return hparams


class Hparams:
    def __init__(self, hparam_name, values):
        self.hparam_name = hparam_name
        self.raw_values = values
        self.values = [Hparam(hparam_name, value) for value in values]

    def __repr__(self):
        return f'{self.hparam_name}: {[value for value in self.raw_values]}\n'


class Hparam:
    def __init__(self, hparam_name, value):
        self.hparam_name = hparam_name
        self.value = value

    def get_command(self):
        return f'--{self.hparam_name} {self.value}'

    def __repr__(self):
        return f'{self.hparam_name}: {self.value}'


class Flags:
    def __init__(self, flag_name, fixed):
        self.flag_name = flag_name
        self.fixed = fixed

        if fixed:
            self.values = [Flag(flag_name, True)]
        else:
            self.values = [Flag(flag_name, True), Flag(flag_name, False)]

    def __repr__(self):
        return f'{self.flag_name}: {"fixed" if self.fixed else "param"}\n'


class Flag:
    def __init__(self, flag_name, present):
        self.flag_name = flag_name
        self.present = present

    def get_command(self):
        if self.present:
            return f'--{self.flag_name}'
        else:
            return ''

    def __repr__(self):
        return f'{self.flag_name}: {self.present}'
