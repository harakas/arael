// Color scheme for the sketch editor canvas.

use eframe::egui;

#[derive(Clone)]
pub struct ColorScheme {
    pub background: egui::Color32,
    pub grid: egui::Color32,
    pub origin: egui::Color32,
    pub line: egui::Color32,
    pub line_selected: egui::Color32,
    pub endpoint: egui::Color32,
    pub endpoint_selected: egui::Color32,
    pub endpoint_line_selected: egui::Color32,
    pub point: egui::Color32,
    pub point_selected: egui::Color32,
    pub point_locked: egui::Color32,
    pub arc: egui::Color32,
    pub constraint_marker: egui::Color32,
    pub preview_line: egui::Color32,
    pub cursor_crosshair: egui::Color32,
    pub status_text: egui::Color32,
}

impl ColorScheme {
    pub fn light() -> Self {
        ColorScheme {
            background: egui::Color32::from_gray(255),
            grid: egui::Color32::from_gray(220),
            origin: egui::Color32::from_rgb(180, 180, 180),
            line: egui::Color32::from_rgb(40, 40, 40),
            line_selected: egui::Color32::from_rgb(255, 140, 0),
            endpoint: egui::Color32::from_rgb(100, 180, 255),
            endpoint_selected: egui::Color32::from_rgb(255, 140, 0),
            endpoint_line_selected: egui::Color32::from_rgb(255, 220, 100),
            point: egui::Color32::from_rgb(100, 180, 255),
            point_selected: egui::Color32::from_rgb(255, 140, 0),
            point_locked: egui::Color32::from_rgb(0, 180, 0),
            arc: egui::Color32::from_rgb(40, 40, 40),
            constraint_marker: egui::Color32::from_rgb(0, 160, 0),
            preview_line: egui::Color32::from_rgba_premultiplied(40, 40, 40, 128),
            cursor_crosshair: egui::Color32::from_rgba_premultiplied(0, 0, 0, 40),
            status_text: egui::Color32::from_gray(80),
        }
    }

    pub fn dark() -> Self {
        ColorScheme {
            background: egui::Color32::from_gray(30),
            grid: egui::Color32::from_gray(45),
            origin: egui::Color32::from_rgb(80, 80, 80),
            line: egui::Color32::from_rgb(200, 200, 200),
            line_selected: egui::Color32::from_rgb(255, 180, 50),
            endpoint: egui::Color32::from_rgb(100, 180, 255),
            endpoint_selected: egui::Color32::from_rgb(255, 180, 50),
            endpoint_line_selected: egui::Color32::from_rgb(255, 220, 100),
            point: egui::Color32::from_rgb(100, 180, 255),
            point_selected: egui::Color32::from_rgb(255, 180, 50),
            point_locked: egui::Color32::from_rgb(0, 200, 0),
            arc: egui::Color32::from_rgb(200, 200, 200),
            constraint_marker: egui::Color32::from_rgb(100, 255, 100),
            preview_line: egui::Color32::from_rgba_premultiplied(200, 200, 200, 128),
            cursor_crosshair: egui::Color32::from_rgba_premultiplied(255, 255, 255, 60),
            status_text: egui::Color32::from_gray(150),
        }
    }
}
