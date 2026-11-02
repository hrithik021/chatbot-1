"""NEXUS RAG — Enterprise SVG Icon System
Lucide-style clean geometry for Light Enterprise UI & Navy Sidebar.
"""

class Icons:
    """Inline SVG icon generator for NEXUS RAG."""

    @staticmethod
    def _svg(path_d: str, size: int = 16, stroke: str = "#667085", sw: float = 1.8, vb: int = 24, fill: str = "none") -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 {vb} {vb}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="display:inline-block;vertical-align:middle;flex-shrink:0">{path_d}</svg>'
        )

    @staticmethod
    def nexus_logo(size: int = 22, color: str = "#2563EB") -> str:
        """Abstract interconnected diamond nodes representing NEXUS intelligence."""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'style="display:inline-block;vertical-align:middle;flex-shrink:0">'
            f'<polygon points="12 2 19 8 12 14 5 8" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
            f'<polygon points="12 10 19 16 12 22 5 16" stroke="{color}" stroke-width="2" stroke-linejoin="round" opacity="0.65"/>'
            f'<circle cx="12" cy="8" r="2" fill="{color}"/>'
            f'<circle cx="12" cy="16" r="2" fill="{color}"/>'
            f'</svg>'
        )

    @staticmethod
    def bot(size: int = 16, color: str = "#2563EB") -> str:
        return Icons._svg(
            '<rect width="18" height="10" x="3" y="11" rx="2"/>'
            '<circle cx="12" cy="5" r="2"/><path d="M12 7v4"/>'
            '<line x1="8" y1="16" x2="8" y2="16"/>'
            '<line x1="16" y1="16" x2="16" y2="16"/>',
            size, color
        )

    @staticmethod
    def user(size: int = 16, color: str = "#2563EB") -> str:
        return Icons._svg(
            '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>'
            '<circle cx="12" cy="7" r="4"/>',
            size, color
        )

    @staticmethod
    def search(size: int = 15, color: str = "#8B9BB4") -> str:
        return Icons._svg(
            '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
            size, color
        )

    @staticmethod
    def plus(size: int = 15, color: str = "#FFFFFF") -> str:
        return Icons._svg('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>', size, color)

    @staticmethod
    def message(size: int = 15, color: str = "#8B9BB4") -> str:
        return Icons._svg('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>', size, color)

    @staticmethod
    def trash(size: int = 14, color: str = "#98A2B3") -> str:
        return Icons._svg(
            '<polyline points="3 6 5 6 21 6"/>'
            '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
            size, color
        )

    @staticmethod
    def upload(size: int = 15, color: str = "#2563EB") -> str:
        return Icons._svg(
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
            '<polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
            size, color
        )

    @staticmethod
    def file_text(size: int = 15, color: str = "#2563EB") -> str:
        return Icons._svg(
            '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>'
            '<polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>'
            '<line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/>',
            size, color
        )

    @staticmethod
    def settings(size: int = 15, color: str = "#667085") -> str:
        return Icons._svg(
            '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>'
            '<line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>'
            '<line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
            '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>'
            '<line x1="17" y1="16" x2="23" y2="16"/>',
            size, color
        )

    @staticmethod
    def send(size: int = 15, color: str = "#FFFFFF") -> str:
        return Icons._svg('<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>', size, color)

    @staticmethod
    def check_circle(size: int = 14, color: str = "#16A34A") -> str:
        return Icons._svg('<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>', size, color)

    @staticmethod
    def alert_triangle(size: int = 14, color: str = "#D97706") -> str:
        return Icons._svg('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', size, color)

    @staticmethod
    def info(size: int = 14, color: str = "#2563EB") -> str:
        return Icons._svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>', size, color)

    @staticmethod
    def zap(size: int = 14, color: str = "#2563EB") -> str:
        return Icons._svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', size, color)

    @staticmethod
    def cpu(size: int = 14, color: str = "#667085") -> str:
        return Icons._svg(
            '<rect width="16" height="16" x="4" y="4" rx="2"/>'
            '<rect width="6" height="6" x="9" y="9"/><path d="M15 2v2"/><path d="M15 20v2"/>'
            '<path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/>'
            '<path d="M9 2v2"/><path d="M9 20v2"/>',
            size, color
        )

    @staticmethod
    def layers(size: int = 14, color: str = "#7C3AED") -> str:
        return Icons._svg('<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>', size, color)

    @staticmethod
    def bookmark(size: int = 13, color: str = "#2563EB") -> str:
        return Icons._svg('<path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/>', size, color)

    @staticmethod
    def dot(size: int = 8, color: str = "#16A34A") -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
            f'style="display:inline-block;vertical-align:middle;flex-shrink:0">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{color}"/></svg>'
        )

    @staticmethod
    def badge_pdf() -> str:
        return (
            '<span style="background:#EF4444;color:#FFFFFF;font-size:0.6rem;font-weight:700;'
            'padding:2px 5px;border-radius:4px;letter-spacing:0.04em">PDF</span>'
        )

    @staticmethod
    def badge_txt() -> str:
        return (
            '<span style="background:#2563EB;color:#FFFFFF;font-size:0.6rem;font-weight:700;'
            'padding:2px 5px;border-radius:4px;letter-spacing:0.04em">TXT</span>'
        )

icons = Icons()
