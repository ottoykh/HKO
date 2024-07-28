import requests
from PIL import Image, UnidentifiedImageError
import numpy as np
import plotly.graph_objs as go
from io import BytesIO
from datetime import datetime, timedelta, timezone

colors_to_extract = {
    ">300": "#ed00f0", 
    "200-300": "#c3006a", 
    "150-200": "#dc0201", 
    "100-150": "#f00000", 
    "75-100": "#ed8202",
    "50-75": "#eeb000", 
    "30-50": "#fada04", 
    "15-30": "#e1cf00", 
    "10-15": "#8fff00", 
    "7-10": "#01f908",
    "5-7": "#01f808", 
    "3-5": "#00d002", 
    "2-3": "#01a835", 
    "1-2": "#008448", 
    "0.50-1": "#3b96ff", 
    "0.15-0.50": "#008ff5"
}

colors_to_extract_rgb = {rate: tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) for rate, color in colors_to_extract.items()}

def extract_color_pixels(img_array, colors, tolerance=30):
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    for color in colors.values():
        color_array = np.array(color)
        diff = np.abs(img_array[..., :3] - color_array)
        mask |= np.all(diff <= tolerance, axis=-1)
    return mask

def count_color_pixels(img_array, colors, tolerance=30):
    counts = []
    for color in colors.values():
        color_array = np.array(color)
        diff = np.abs(img_array[..., :3] - color_array)
        mask = np.all(diff <= tolerance, axis=-1)
        counts.append(np.sum(mask))
    return counts

def process_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGBA")
        width, height = img.size
        left = (width - 320) / 2
        top = (height - 400) / 2
        right = (width + 320) / 2
        bottom = (height + 400) / 2
        img_cropped = img.crop((left, top, right, bottom))
        
        img_array = np.array(img_cropped)
        color_counts = count_color_pixels(img_array, colors_to_extract_rgb)
        return color_counts
    except (requests.exceptions.RequestException, UnidentifiedImageError) as e:
        print(f"Error processing image from {url}: {e}")
        return None

def plot_timeline():
    hkt = timezone(timedelta(hours=8))
    now = datetime.now(hkt)

    intervals = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
    timestamps = []
    for hour_offset in range(3):
        hour = (now - timedelta(hours=hour_offset)).replace(minute=0, second=0, microsecond=0)
        for minute in intervals:
            timestamp = hour + timedelta(minutes=minute)
            if timestamp <= now:
                timestamps.append(timestamp)
    timestamps.sort(reverse=True)

    base_url = "https://www.hko.gov.hk/wxinfo/radars/rad_064_png/2d064nradar_"
    urls = [base_url + timestamp.strftime("%Y%m%d%H%M") + ".jpg" for timestamp in timestamps]

    color_counts_over_time = [process_image(url) for url in urls]
    valid_color_counts = [counts for counts in color_counts_over_time if counts is not None]
    valid_timestamps = [timestamp for counts, timestamp in zip(color_counts_over_time, timestamps) if counts is not None]
    valid_color_counts = np.array(valid_color_counts)

    data = []
    for i, (rate, color) in enumerate(colors_to_extract.items()):
        area_values = valid_color_counts[:, i] / 1765.2936237
        filtered_area_values = [value if value >= 0.45 else 0 for value in area_values]
        trace = go.Scatter(
            x=valid_timestamps,
            y=filtered_area_values,
            mode='lines',
            name=f'{rate} mm/hr',
            line=dict(color=color)
        )
        data.append(trace)

    layout = go.Layout(
        title="Rain Rate (mm/h) Trends Over Time",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Square Kilometre of Raining Area"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="right",
            x=1,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(xaxis=dict(tickformat="%H:%M"))
    fig.show()
