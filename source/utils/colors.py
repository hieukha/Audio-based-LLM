# colors.py - Terminal color utilities
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    ORANGE = "\033[38;5;208m"
    PINK = "\033[38;5;213m"
    LIGHT_GRAY = "\033[37m"

    @classmethod
    def apply(cls, text: str):
        """Returns a ColoredText object for chaining color methods."""
        return ColoredText(text)


class ColoredText:
    """Helper class for applying colors to text."""
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def red(self) -> str:
        return f"{Colors.RED}{self.text}{Colors.RESET}"

    def green(self) -> str:
        return f"{Colors.GREEN}{self.text}{Colors.RESET}"

    def yellow(self) -> str:
        return f"{Colors.YELLOW}{self.text}{Colors.RESET}"

    def blue(self) -> str:
        return f"{Colors.BLUE}{self.text}{Colors.RESET}"

    def magenta(self) -> str:
        return f"{Colors.MAGENTA}{self.text}{Colors.RESET}"

    def cyan(self) -> str:
        return f"{Colors.CYAN}{self.text}{Colors.RESET}"

    def white(self) -> str:
        return f"{Colors.WHITE}{self.text}{Colors.RESET}"

    def gray(self) -> str:
        return f"{Colors.GRAY}{self.text}{Colors.RESET}"

    def orange(self) -> str:
        return f"{Colors.ORANGE}{self.text}{Colors.RESET}"

    def pink(self) -> str:
        return f"{Colors.PINK}{self.text}{Colors.RESET}"

    def light_gray(self) -> str:
        return f"{Colors.LIGHT_GRAY}{self.text}{Colors.RESET}"

