"""
Console Output Manager.
Handles rich console output, progress bars, and status tables.
"""
from datetime import datetime, timedelta

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()

class PipelineConsole:
    """
    Manages the live CLI dashboard using the Rich library.
    
    Provides real-time visualization of processing progress, metrics, and
    activity logs in a structured layout.
    """
    def __init__(self):
        """Initializes progress bars, stats counters, and layout state."""
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="white"),
            TextColumn("[white]{task.description}"),
            BarColumn(bar_width=None, complete_style="white", finished_style="white"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•", TimeElapsedColumn(), "•", TimeRemainingColumn(),
            expand=True
        )
        self.overall_task = None
        self.ocr_task = None
        self.stats = {"processed": 0, "success": 0, "failed": 0, "ocr_done": 0, "ocr_total": 0, "start_time": datetime.now(), "ocr_start_time": None, "total_pages": 0}
        self.logs = []
        self.live = None

    def _generate_layout(self) -> Layout:
        """
        Builds the dashboard layout split into header, body (main+stats).
        
        Returns:
            A Rich Layout object populated with current state.
        """
        layout = Layout()
        layout.split(Layout(name="header", size=3), Layout(name="body", ratio=1))
        layout["body"].split_row(Layout(name="main", ratio=2), Layout(name="stats", ratio=1))
        duration_str = str(datetime.now() - self.stats["start_time"]).split(".")[0]
        header_text = f"[bold white]NASA TRANSCRIPT PIPELINE[/bold white] [dim]|[/dim] [white]STATUS: RUNNING[/white] [dim]|[/dim] [white]TIME: {duration_str}[/white]"
        layout["header"].update(Panel(header_text, style="white", box=box.ROUNDED))
        log_panel = Panel("\n".join(self.logs[-14:]), title="Activity Log", border_style="white", box=box.ROUNDED, height=18)
        layout["main"].update(Group(Panel(self.progress, title="Progress", border_style="white", box=box.ROUNDED), log_panel))
        stats_table = Table(box=box.SIMPLE, expand=True, show_header=False)
        stats_table.add_column("Metric", style="dim white")
        stats_table.add_column("Value", style="bold white", justify="right")
        processed = self.stats["processed"]
        total_pages = self.stats["total_pages"]
        ocr_done = self.stats["ocr_done"]
        ocr_total = self.stats["ocr_total"]
        failed = self.stats["failed"]
        stats_table.add_row("Total Pages", str(total_pages))
        stats_table.add_row("Processed", f"{processed}/{total_pages}")
        stats_table.add_row("OCR Progress", f"{ocr_done}/{ocr_total}" if ocr_total else "-")
        stats_table.add_row("Failures", f"[red]{failed}[/red]" if failed > 0 else "0")
        if ocr_done > 0 and ocr_total > 0 and self.stats["ocr_start_time"]:
            elapsed = (datetime.now() - self.stats["ocr_start_time"]).total_seconds()
            if elapsed > 0:
                speed = self.stats["ocr_done"] / elapsed
                eta = str(timedelta(seconds=int((self.stats["ocr_total"] - self.stats["ocr_done"]) / speed))).split(".")[0]
                stats_table.add_section()
                stats_table.add_row("OCR Speed", f"{speed:.2f} p/s")
                stats_table.add_row("OCR ETA", f"~{eta}")
        layout["stats"].update(Panel(stats_table, title="Metrics", border_style="white", box=box.ROUNDED))
        return layout

    def start_pipeline(self, total_pages: int, pdf_name: str):
        """
        Activates the live display and starts the timer.

        Args:
            total_pages: Total number of pages to process.
            pdf_name: Name of the PDF being processed.
        """
        self.stats["total_pages"] = total_pages
        self.stats["start_time"] = datetime.now()
        self.overall_task = self.progress.add_task("Image Processing", total=total_pages)
        self.live = Live(self._generate_layout(), refresh_per_second=4, console=console, screen=True)
        self.live.start()

    def log(self, message: str):
        """
        Adds a timestamped message to the activity log panel.

        Args:
            message: Text or markup string to log.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[dim]{timestamp}[/dim] {message}")
        if self.live:
            self.live.update(self._generate_layout())

    def update_image_progress(self, advance: int = 1):
        """
        Increments the image processing progress bar.

        Args:
            advance: Number of pages completed.
        """
        self.progress.update(self.overall_task, advance=advance)
        self.stats["processed"] += advance
        self.stats["success"] += advance
        if self.live:
            self.live.update(self._generate_layout())

    def start_ocr(self, total_to_ocr: int):
        """
        Initializes the OCR progress bar.

        Args:
            total_to_ocr: Number of successful pages ready for OCR.
        """
        self.stats["ocr_total"] = total_to_ocr
        self.stats["ocr_start_time"] = datetime.now()
        self.ocr_task = self.progress.add_task("OCR Intelligence", total=total_to_ocr)
        self.log("[white]Initializing OCR Phase...[/white]")

    def update_ocr_progress(self, page_num: int, duration: float, timing_info: str | None = None):
        """
        Increments the OCR progress bar and logs completion time.

        Args:
            page_num: Zero-indexed page completed.
            duration: Time taken in seconds.
            timing_info: Optional detailed timing breakdown.
        """
        self.progress.update(self.ocr_task, advance=1)
        self.stats["ocr_done"] += 1
        timing_suffix = f" | {timing_info}" if timing_info else ""
        self.log(f"Page {page_num+1:<3} [green]Done[/green] ({duration:.1f}s){timing_suffix}")

    def fail_ocr(self, page_num: int, error: str):
        """
        Logs an OCR failure and increments the error counter.

        Args:
            page_num: Page that failed.
            error: Error message string.
        """
        self.stats["failed"] += 1
        self.log(f"Page {page_num+1:<3} [red]Error[/red]: {error}")

    def finish(self):
        """
        Stops the live display and prints a final session summary table.
        """
        if self.live:
            self.live.stop()
        duration_str = str(datetime.now() - self.stats["start_time"]).split(".")[0]
        console.print("\n")
        console.rule("[bold white]MISSION ACCOMPLISHED")
        summary = Table(box=box.ROUNDED, show_header=False)
        summary.add_column("Metric", style="dim white")
        summary.add_column("Value", style="bold white")
        summary.add_row("Total Duration", duration_str)
        summary.add_row("Pages Processed", str(self.stats["processed"]))
        ocr_done = self.stats["ocr_done"]
        failed = self.stats["failed"]
        if ocr_done > 0:
            summary.add_row("OCR Completed", str(ocr_done))
        summary.add_row("Failures", f"[red]{failed}[/red]" if failed else "[green]0[/green]")
        summary.add_row("Status", "[green]SUCCESS[/green]" if not failed else "[red]FAILED[/red]")
        console.print(summary)
        console.print("\n")
