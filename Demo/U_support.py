from RASP_support.REPL import REPL

class REPLwithU(REPL):
    def __init__(self):
        super().__init__()
        self.load_libraries_for_U()

    def load_libraries_for_U(self):
        self.silent = True
        # base env: the env from which every load begins
        self.base_env = self.env.snapshot()
        # bootstrap base_env with current (basically empty except indices etc)
        # env, then load the base libraries to build the actual base env
        # make the library-loaded variables and functions not-overwriteable
        self.env.storing_in_constants = True
        for lib in ["U/Ulib"]:
            self.run_given_line("load \"" + lib + "\";")
            self.base_env = self.env.snapshot()
        self.env.storing_in_constants = False
        self.run_given_line("tokens=tokens_float;")
        self.base_env = self.env.snapshot()
        self.silent = False
