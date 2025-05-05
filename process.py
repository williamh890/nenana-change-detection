import json
import os

from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import pandas as pd
from osgeo import gdal
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
import matplotlib.patches as patches
from matplotlib import cm, colors


gdal.UseExceptions()
PRODUCT_PATH = Path.cwd() / 'products'
HYP3_PRODUCT_PATH = Path.cwd() / 'hyp3-products'

AIRPORT_SUBSET = [464, 461, 4, 4]
NENANA_PROPER_SUBSET = [435, 398, 37, 40]
QUARY_SUBSET = [425, 330, 12, 12]
NENANA_FULL = [375, 320, 180, 180]


def process_bursts():
    with open('bursts.json', 'r') as f:
        bursts = json.load(f)['bursts']

    for burst in bursts:
        os.system(f'./run.sh {burst}')


def make_pandas_datetime_index():

    dates = []
    for burst_path in PRODUCT_PATH.glob('*-BURST'):
        date_str = str(burst_path).split('_')[3].split('T')[0]
        date = datetime.strptime(date_str, '%Y%m%d').date()

        dates.append(date)

    time_index = pd.DatetimeIndex(dates)

    return time_index


def get_burst_date(burst_path):
    date_str = str(burst_path).split('_')[3].split('T')[0]
    return datetime.strptime(date_str, '%Y%m%d').date()


def log_diff():
    # burst_paths = list(PRODUCT_PATH.glob('*-BURST'))
    # burst_paths.sort(key=get_burst_date)
    # datasets = [open_burst(burst_path, '*_VV.tif') for burst_path in burst_paths]
    subset = NENANA_FULL

    def get_burst_date(burst_path):
        date_str = str(burst_path.name).split('_')[4].split('T')[0]
        return datetime.strptime(date_str, '%Y%m%d').date()

    burst_paths = [tif_path for tif_path in HYP3_PRODUCT_PATH.glob('*_VV.tif')]
    burst_paths.sort(key=get_burst_date)
    datasets = [gdal.Open(path, gdal.GA_ReadOnly) for path in burst_paths]

    data, dates, date_text = [], [], []

    for idx, burst_path in enumerate(burst_paths[:-1]):
        print(f'loading {burst_path.name}')
        dataset, next_dataset = datasets[idx], datasets[idx + 1]

        if dataset is None or next_dataset is None:
            continue

        power_arr, next_power = get_array(dataset), get_array(next_dataset)
        date_range = (get_burst_date(burst_path), get_burst_date(burst_paths[idx + 1]))

        log_diff = np.log10(next_power / power_arr)
        subset_data = log_diff[subset[1]:(subset[1] + subset[3]), subset[0]:(subset[0] + subset[2])]
        data.append(subset_data)

        dates.append(date_range[0])
        date_text.append(date_range)

    full_stack = np.stack(data)
    print(np.max(full_stack), np.min(full_stack))
    norm = colors.Normalize(vmin=-3, vmax=3, clip=False)
    time_index = pd.DatetimeIndex(dates)
    # ts = pd.Series(data, index=time_index)

    save_gif(data, date_text)
    #show_image(data, time_index, 1, subset=NENANA_FULL)


def save_gif(data, date_ranges):
    norm = colors.Normalize(vmin=-2, vmax=2, clip=False)
    imgs = [Image.fromarray(
        np.uint8(cm.Spectral_r(norm(arr)) * 255)
    ) for arr in data]

    for img, date_range in zip(imgs, date_ranges):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("sans-serif.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        start, end = date_range
        date_text = f'{start.strftime("%x")} - {end.strftime("%x")}'
        draw.text((5, 162), date_text, (0, 0, 0), font=font)
        img.save(f'log_diffs/log_diff_{start.strftime("%Y%m%d")}-{end.strftime("%Y%m%d")}.png')

    imgs[0].save("log_diff.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0)


def make_hyp3_product_timeseries():
    subset = NENANA_FULL

    def get_date(burst_path):
        date_str = str(burst_path.name).split('_')[4].split('T')[0]
        return datetime.strptime(date_str, '%Y%m%d').date()

    tif_paths = [tif_path for tif_path in HYP3_PRODUCT_PATH.glob('*_VV.tif')]
    tif_paths.sort(key=get_date)
    print(tif_paths[0].name)
    print(tif_paths[-1].name)

    data, dates = [], []
    for tif_path in tif_paths:
        print(f'loading {tif_path.name}')
        dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if dataset is None:
            continue
        power_arr = get_array(dataset)

        subset_data = power_arr[subset[1]:(subset[1] + subset[3]), subset[0]:(subset[0] + subset[2])]
        data.append(subset_data)

        dates.append(get_date(tif_path))

    print(np.uint8(cm.Greys(subset_data) * 255))
    data = np.stack(data)
    data_db = 10 * np.log10(data)
    # time_index = pd.DatetimeIndex(dates)
    # ts = pd.Series(data, index=time_index)

    imgs = [Image.fromarray(
        np.uint8(cm.grey(arr) * 255)
    ) for arr in data]

    for img, date in zip(imgs, dates):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("sans-serif.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        date_text = f'{date.strftime("%x")}'
        draw.text((5, 162), date_text, (255, 255, 255), font=font)
        img.save(f'db_backscatter/db_backscatter_{date.strftime("%Y%m%d")}.png')

    imgs[0].save("db_backscatter.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0)


def make_product_timeseries():
    burst_paths = list(PRODUCT_PATH.glob('*-BURST'))
    burst_paths.sort(key=get_burst_date)

    data, dates = [], []
    for burst_path in burst_paths:
        print(f'loading {burst_path.name}')
        dataset = open_burst(burst_path, '*_VH.tif')
        if dataset is None:
            continue
        power_arr = get_array(dataset)

        data.append(power_arr)
        dates.append(get_burst_date(burst_path))

    data = np.stack(data)
    data_db = 10 * np.log10(data)
    time_index = pd.DatetimeIndex(dates)
    # ts = pd.Series(data, index=time_index)

    print('QUARY SUBSET')
    show_image(data_db, time_index, 1, subset=QUARY_SUBSET)


def show_image(raster_stack, time_index, band_number, subset=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(raster_stack[band_number - 1], cmap='gray')
    ax1.set_title('Image Band {} {}'.format(band_number, time_index[band_number - 1].date()))
    if subset is None:
        bands, ydim, xdim = raster_stack.shape
        subset = (0, 0, xdim, ydim)

    ax1.add_patch(patches.Rectangle((subset[0], subset[1]), subset[2], subset[3], fill=False, edgecolor='red'))
    ax1.xaxis.set_label_text('Pixel')
    ax1.yaxis.set_label_text('Line')

    ts_db = np.nanmean(raster_stack[:, subset[1]:(subset[1] + subset[3]), subset[0]:(subset[0] + subset[2])], axis=(1,2))
    print(ts_db)
    ax2.plot(time_index, ts_db)
    ax2.yaxis.set_label_text('$\\gamma^o$ [dB]')
    ax2.set_title('$\\gamma^o$ Backscatter Time Series')
    # Add a vertical line for the date where the image is displayed
    ax2.axvline(time_index[band_number - 1], color='red')
    plt.grid()

    fig.autofmt_xdate()
    plt.show()


def basic_plot(ts):
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(16, 8))
    plt.title("Time Series of Means")
    ts.plot()
    plt.xlabel('Date')
    plt.ylabel('$\\overline{\\gamma^o}$ [dB]')
    plt.grid()
    plt.savefig('time_series_means.png', dpi=72)
    plt.show()


def write_gtiff(source, output_data, output_path):
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = source.shape

    print(output_path)
    outdata = driver.Create(str(output_path), cols, rows, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform(source.GetGeoTransform())
    outdata.SetProjection(source.GetProjection())
    outdata.GetRasterBand(1).WriteArray(output_data)
    outdata.FlushCache()


def get_array(vv_ds):
    vv_rb = vv_ds.GetRasterBand(1)
    return vv_rb.ReadAsArray()


def open_burst(burst_path, file_extension):
    try:
        path = next(burst_path.glob(file_extension))
    except StopIteration:
        return None
    else:
        return gdal.Open(path, gdal.GA_ReadOnly)


if __name__ == '__main__':
    log_diff()
    # make_product_timeseries()
    # make_hyp3_product_timeseries()
