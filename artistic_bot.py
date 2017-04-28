import logging
import os
import numpy as np
import json
import random
import argparse

from enum import Enum
from io import BytesIO
from PIL import Image
from multiprocessing import Process
from multiprocessing import Queue

import telegram
from telegram.ext import Updater
from telegram.ext import ConversationHandler
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import Filters

from artistic_style import transfer_style


dirname = os.path.abspath(os.path.dirname(__file__))
STYLES_ROOT = os.path.join(dirname, 'data/styles')
STYLES = os.listdir(STYLES_ROOT)
DEVICE = '/cpu:0'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARN)
logger = logging.getLogger(__name__)


WELCOME_MESSAGE = '''Welcome to the Artistic Style Bot!
Send me a photo with a style in caption to begin. /help
'''

HELP_MESSAGE = '''I can turn any of your photos into an artwork.
Just send me a photo with a style name in a caption to the photo.

Commands:

/liststyles - List all available styles
/showstyle - Show original style artwork
'''

REPLY_CAPTIONS = [
    'Here we go... Now it looks much better!',
    'Wow, it was really hard...',
    'DONE',
    'DONE!!',
    "- Knock-knock.\n- Who's there?\n- A photo.\n\njk"
]

PHOTO_REPLIES = [
    "I will take care of this photo as soon as possible, don't worry. :)",
    "Working hardly... Be patient!"
]


job_queue = Queue()


def load_image(filename):
    image = Image.open(filename)
    image = np.asarray(image, np.float32)
    return image


def download_image(bot, photo):
    buf = BytesIO()
    f = bot.get_file(photo.file_id)
    f.download(out=buf)
    image = load_image(buf)
    return image


def convert_image_to_buf(image):
    buf = BytesIO()
    buf.name = 'image.png'
    image = image.astype(np.uint8)
    image = Image.fromarray(image, 'RGB')
    image.save(buf, 'PNG')
    buf.seek(0)
    return buf


def worker(bot, queue, device):
    """
    TODO: make an input pipeline which handles downloading
          images from Telegram server.
    """
    while True:
        message = queue.get()
        style = message.caption
        photo = message.photo[-1]
        if not style:
            message.reply_text('You forgot style name. /liststyles')
            continue
        style = style.strip().lower()
        if style not in STYLES:
            message.reply_text("Sorry, I don't have such a style. /liststyles")
            continue
        image = download_image(bot, photo)
        model_filename = os.path.join(STYLES_ROOT, style, 'model')
        out_image = transfer_style(image, model_filename, device)
        out_image_buf = convert_image_to_buf(out_image)
        caption = random.choice(REPLY_CAPTIONS)
        message.reply_photo(out_image_buf, caption=caption)


def start(bot, update):
    logger.info('bot started')
    bot.send_message(chat_id=update.message.chat_id,
                     text=WELCOME_MESSAGE)


def photo(bot, update):
    message = update.message
    reply = random.choice(PHOTO_REPLIES)
    message.reply_text(reply)
    job_queue.put(message)


def list_styles(bot, update):
    text = '\n'.join('- ' + style for style in STYLES)
    update.message.reply_text(text)


def show_style(bot, update, args):
    args = filter(lambda style: style in STYLES, args)
    for style in args:
        photo_filename = os.path.join(STYLES_ROOT, style, 'original.jpg')
        info_filename = os.path.join(STYLES_ROOT, style, 'info')
        image = load_image(photo_filename)
        buf = convert_image_to_buf(image)
        with open(info_filename, 'r') as f:
            info = json.load(f)
        caption = '{name}\nby {artist}, {year}.'.format(**info)
        update.message.reply_photo(buf, caption=caption)


def help(bot, update):
    update.message.reply_text(HELP_MESSAGE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',
                        help='compute device',
                        default=DEVICE)
    parser.add_argument('--token',
                        help='authentication token',
                        required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    updater = Updater(args.token)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    photo_handler = MessageHandler(Filters.photo, photo)
    dispatcher.add_handler(photo_handler)

    liststyles_handler = CommandHandler('liststyles', list_styles)
    dispatcher.add_handler(liststyles_handler)

    showstyle_handler = CommandHandler('showstyle', show_style, pass_args=True)
    dispatcher.add_handler(showstyle_handler)

    help_handler = CommandHandler('help', help)
    dispatcher.add_handler(help_handler)

    # Spawn worker process. Worker process is mostly doing processor-intensive
    # tasks and this process is handling only IO stuff.
    worker_args = (updater.bot, job_queue, args.device)
    worker_process = Process(target=worker, args=worker_args)
    worker_process.start()

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
