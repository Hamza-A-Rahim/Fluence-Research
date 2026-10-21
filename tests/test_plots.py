"""Tests for inspection figures."""

import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fluence.visualization.plots import make_inspection_figure, plot_waveform_and_logmel


class TestInspectionFigure(unittest.TestCase):
    def setUp(self):
        self.x = (0.3 * np.sin(2 * np.pi * 200 * np.arange(32000) / 16000)).astype(np.float32)

    def test_time_axes_line_up(self):
        # Regression test: a colour bar attached to one panel once made the two
        # panels different widths, so the same time sat at different x positions.
        fig = make_inspection_figure(self.x, 16000, spans=[(0.5, 1.0, "event")])
        wave, spec = fig.axes[0], fig.axes[1]
        fig.canvas.draw()
        pw, ps = wave.get_position(), spec.get_position()
        self.assertAlmostEqual(pw.x0, ps.x0, places=4)
        self.assertAlmostEqual(pw.x1, ps.x1, places=4)
        self.assertEqual(wave.get_xlim(), spec.get_xlim())
        plt.close(fig)

    def test_writes_png(self):
        with tempfile.TemporaryDirectory() as d:
            out = plot_waveform_and_logmel(self.x, 16000, Path(d) / "f.png", title="t")
            self.assertTrue(out.is_file())
            self.assertEqual(out.read_bytes()[:4], b"\x89PNG")


if __name__ == "__main__":
    unittest.main()
