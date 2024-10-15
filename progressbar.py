import time

class ProgressBar:
    def __init__(self, total, length=50, fill='█', print_end='\r', label='Progress'):
        """
        Initialize the progress bar.
        :param total: Total iterations (int)
        :param length: Character length of bar (int)
        :param fill: Bar fill character (str)
        :param print_end: End character (e.g. '\r' or '\n') (str)
        """
        self.total = total
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.iteration = 0
        self.label = label

    def print_progress_bar(self):
        """
        Print the progress bar to the console.
        """
        percent = ("{0:.1f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.label} : {percent}% |{bar}| {self.iteration}/{self.total}', end=self.print_end)
        if self.iteration == self.total:
            print()  # Print a new line on complete

    def update(self, step=1):
        """
        Update the progress bar by a given step.
        :param step: Step size (int)
        """
        self.iteration += step
        self.print_progress_bar()

    def complete(self):
        """
        Mark the progress as complete.
        """
        self.iteration = self.total
        self.print_progress_bar()

# Example usage:
def example_usage():
    total_items = 100
    progress_bar = ProgressBar(total_items, label='Recovering labels')

    for i in range(total_items):
        time.sleep(0.1)  # Simulate work being done
        progress_bar.update()

    progress_bar.complete()

if __name__ == "__main__":
    example_usage()