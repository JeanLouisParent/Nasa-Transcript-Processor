"""
Console Output Manager.

Handles rich console output, progress bars, and status tables for the pipeline.
"""

from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.layout import Layout
from rich import box

console = Console()

class PipelineConsole:
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console
        )
        self.overall_task: Optional[TaskID] = None
        self.ocr_task: Optional[TaskID] = None
        self.layout: Layout = Layout()
        self.stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "ocr_done": 0,
            "start_time": datetime.now()
        }

    def start_pipeline(self, total_pages: int, pdf_name: str):
        """Initialize the layout and start the main progress bar."""
        self.stats["total"] = total_pages
        self.stats["start_time"] = datetime.now()
        
        console.print(Panel(
            f"[bold green]NASA Transcript Pipeline v2.0[/bold green]\n"
            f"Processing: [cyan]{pdf_name}[/cyan]\n"
            f"Total Pages: [bold]{total_pages}[/bold]",
            box=box.DOUBLE,
            title="Mission Control",
            border_style="blue"
        ))
        
        self.progress.start()
        self.overall_task = self.progress.add_task(
            "[cyan]Image Processing...", 
            total=total_pages
        )

    def update_image_progress(self, advance: int = 1):
        """Update the image processing progress bar."""
        self.progress.update(self.overall_task, advance=advance)
        self.stats["processed"] += advance
        self.stats["success"] += advance # Assuming success for now in the bar

    def start_ocr(self, total_to_ocr: int):
        """Add a secondary progress bar for OCR."""
        self.ocr_task = self.progress.add_task(
            "[magenta]OCR Processing...", 
            total=total_to_ocr
        )

    def update_ocr_progress(self, page_num: int, duration: float):
        """Update OCR progress and log specific page info."""
        self.progress.update(self.ocr_task, advance=1)
        self.stats["ocr_done"] += 1
        
        # We can print a log line above the progress bar
        console.print(
            f"  [green]✓[/green] Page {page_num+1:<4} "
            f"[dim]OCR: {duration:.1f}s[/dim]"
        )

    def fail_ocr(self, page_num: int, error: str):
        """Log an OCR failure."""
        console.print(
            f"  [red]✗[/red] Page {page_num+1:<4} "
            f"[red]Failed: {error}[/red]"
        )
        self.stats["failed"] += 1

    def finish(self):
        """Stop the progress bars and print summary."""
        self.progress.stop()
        
        duration = datetime.now() - self.stats["start_time"]
        
        summary = Table(title="Mission Summary", box=box.ROUNDED)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="bold white")
        
        summary.add_row("Total Time", str(duration).split('.')[0])
        summary.add_row("Pages Processed", str(self.stats["processed"]))
        summary.add_row("OCR Completed", str(self.stats["ocr_done"]))
        
        if self.stats["failed"] > 0:
            summary.add_row("Failures", f"[red]{self.stats['failed']}[/red]")
        else:
            summary.add_row("Status", "[green]SUCCESS[/green]")

        console.print(summary)