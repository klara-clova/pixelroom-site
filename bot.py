"""
PixelRoom - image description bot for blind and visually impaired Discord users.

The bot stays quiet and answers only when someone asks. Every answer is
private (only the person who asked can see it).

Commands
  /describe [back] [count] [detail]  describe the latest image(s) in the channel
  /ask question [back]               ask a question about an image
  /readtext [back]                   read out all the text in an image
  Right click a message > Apps > Describe image

Images, GIFs and short video clips are supported. Answers come in the
language the user's Discord is set to.
"""

import asyncio
import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageSequence

# ---------- Settings ----------

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

# Automatic mode: describe every new image publicly in the channel.
# Off = the bot stays quiet and only answers when someone asks.
AUTO_DESCRIBE = False

# How long to wait for the AI before trying again (videos get a bit longer).
# Sometimes a request gets stuck at Google; a quick retry almost always works.
AI_TIMEOUT = 15
AI_TIMEOUT_VIDEO = 25
AI_ATTEMPTS = 3

# File types the bot understands
STILL_TYPES = {"image/png", "image/jpeg", "image/webp"}
GIF_TYPE = "image/gif"
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}

# Other video formats (MKV from OBS, AVI, WMV...) are converted to MP4 first.
# This needs ffmpeg on the server: apt install -y ffmpeg
FFMPEG = shutil.which("ffmpeg")
CONVERT_EXTENSIONS = {".mkv", ".avi", ".wmv", ".flv", ".m4v", ".3gp", ".mpg", ".mpeg", ".ts"}
# Only the first part of a long video is sent to the AI, to keep it fast and cheap
MAX_VIDEO_SECONDS = 60

# Files people post that look like images or videos, used to warn the user
# when we have to skip one we can't read
MEDIA_EXTENSIONS = CONVERT_EXTENSIONS | {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov",
    ".heic", ".heif", ".bmp", ".tiff", ".tif", ".svg", ".avif",
}

# How many frames we take from an animated GIF
GIF_FRAMES = 3

# Skip very large files (in bytes) - 15 MB
MAX_FILE_SIZE = 15 * 1024 * 1024

# How many images the bot works on at the same time
MAX_PARALLEL = 3

# How far back /describe, /ask and /readtext look for images
HISTORY_LIMIT = 100

# Discord message length limit
DISCORD_LIMIT = 2000

# Discord language codes -> language names for the AI
LANGUAGES = {
    "id": "Indonesian", "da": "Danish", "de": "German", "en-GB": "English",
    "en-US": "English", "es-ES": "Spanish", "es-419": "Spanish", "fr": "French",
    "hr": "Croatian", "it": "Italian", "lt": "Lithuanian", "hu": "Hungarian",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt-BR": "Portuguese",
    "ro": "Romanian", "fi": "Finnish", "sv-SE": "Swedish", "vi": "Vietnamese",
    "tr": "Turkish", "cs": "Czech", "el": "Greek", "bg": "Bulgarian",
    "ru": "Russian", "uk": "Ukrainian", "hi": "Hindi", "th": "Thai",
    "zh-CN": "Chinese (Simplified)", "ja": "Japanese", "zh-TW": "Chinese (Traditional)",
    "ko": "Korean",
}

# ---------- What we tell the AI ----------

BASE_RULES = (
    "You are helping a blind or visually impaired person who uses a screen "
    "reader on Discord.\n"
    "General rules:\n"
    "- Plain text only. No markdown, no bullet points, no emojis.\n"
    "- Never identify or name real people from their face or appearance, even "
    "if they look famous. Describe them instead, for example 'a smiling man in "
    "a baseball cap'. Only use a real person's name if it is written in the "
    "image itself. Fictional characters from games, films and cartoons may be "
    "named.\n"
)

SHORT_RULES = (
    "Task: describe the image.\n"
    "- Start directly with the content. Never start with 'This image shows' "
    "or 'The image is'.\n"
    "- Keep it short: 1 to 3 sentences for simple images. Use more only if the "
    "image contains a lot of important information.\n"
    "- Describe what matters: people, characters, actions, mood, and the main "
    "point of the image. Skip tiny unimportant details.\n"
    "- If it is a game or app screenshot, focus on the scene, the characters "
    "and what is happening. Do not list interface elements (health bars, "
    "minimaps, compasses, hotbars, clocks, chat hints). Mention the interface "
    "only if it is clearly the point, for example an inventory, a reward, a "
    "quest or an error message.\n"
    "- If there is meaningful readable text (signs, captions, memes, posters), "
    "write it out exactly after the description, starting with the label "
    "'Text in image:'. Skip interface labels, clock times and hints like "
    "'Press Enter to chat'. If there is no meaningful text, leave the label out.\n"
    "- If the image contains a lot of text (a map, a menu, a legend, a long "
    "list or a long document), don't write it all out. Give the title and the "
    "most important few items, then say that /readtext reads out all of it "
    "(in the answer language, keeping '/readtext' as it is). In that case do "
    "NOT add the 'Text in image:' section at all.\n"
)

LONG_RULES = (
    "Task: give a thorough, detailed description of the image.\n"
    "- Start directly with the content. Never start with 'This image shows'.\n"
    "- Begin with one sentence that gives the overall picture, then go into "
    "detail: the layout from top to bottom or left to right, people or "
    "characters (appearance, clothes, expressions, actions), objects, colours, "
    "lighting, background and mood.\n"
    "- For screenshots, also describe the important interface elements and "
    "what they show.\n"
    "- Write out all readable text exactly, starting with the label "
    "'Text in image:'.\n"
)

TEXT_RULES = (
    "Task: read out all the meaningful text in the image.\n"
    "- Write the text exactly as it appears, in reading order (top to bottom, "
    "left to right). Do not translate it, do not summarise it, do not describe "
    "the picture.\n"
    "- Keep separate blocks of text (a title, a paragraph, a list) on separate "
    "lines.\n"
    "- Put short labels that belong together on ONE line, separated by commas, "
    "for example all the place names on a map, all the items in a legend, or "
    "all the buttons in a menu.\n"
    "- Leave out text that carries no meaning on its own: grid letters and "
    "numbers along the edges of a map or chart, single stray letters or "
    "digits, page numbers, watermarks, and control hints like 'Press E' or "
    "'Look Around'.\n"
    "- If the image contains no readable text, say so in one short sentence, "
    "then add: 'Use /describe to hear what the image shows.' (translated into "
    "the answer language, but keep '/describe' as it is).\n"
)

ASK_RULES = (
    "Task: answer the user's question about the image.\n"
    "- Answer the actual question first, directly, in 1 to 3 sentences, "
    "unless the question needs a longer answer. Don't describe the whole "
    "image unless that is what was asked.\n"
    "- If the user asks which game, film, show, app, place or character this "
    "is, and you can recognise it from the image (its art style, place names, "
    "logos or interface), name it. If you are only fairly sure, say so, for "
    "example 'This looks like Palia'. Never identify real people.\n"
    "- Only say what you can actually see. If the image does not show the "
    "answer, or you are not sure, say so honestly.\n"
    "- Do not add a separate section with the text from the image. Quote text "
    "from the image only if it helps answer the question.\n"
    "The user's question is: "
)

ANIMATION_NOTE = (
    "\nWhat you receive is ONE animated GIF or video clip (as a video, or as "
    "frames in order). Treat it as one animation: who or what is in it and "
    "what happens or moves. Do not describe frames one by one. Ignore sound."
)


LABELS = {
    "English": "Text in image", "Hungarian": "Szöveg a képen",
    "German": "Text im Bild", "French": "Texte dans l'image",
    "Spanish": "Texto en la imagen",
}


def strip_labels(text: str) -> str:
    """Remove 'Text in image:' style labels the AI may add in front of lines."""
    prefixes = [f"{label}:" for label in LABELS.values()]
    cleaned = []
    for line in text.splitlines():
        stripped = line.lstrip()
        for prefix in prefixes:
            if stripped.lower().startswith(prefix.lower()):
                stripped = stripped[len(prefix):].lstrip()
                break
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


# A short description should never read out a whole wall of text
MAX_TEXT_SECTION = 220


def trim_text_section(text: str, language: str) -> str:
    """Shorten an overly long 'Text in image:' section and point to /readtext."""
    labels = [f"{LABELS[language]}:"] if language in LABELS else []
    labels.append("Text in image:")
    lower = text.lower()
    for label in labels:
        pos = lower.find(label.lower())
        if pos == -1:
            continue
        section = text[pos + len(label):].strip()
        if len(section) <= MAX_TEXT_SECTION:
            return text
        cut = section[:MAX_TEXT_SECTION].rsplit(" ", 1)[0].rstrip(",;:")
        before = text[:pos].rstrip()
        return f"{before}\n\n{label} {cut}... {tr(language, 'readtext_hint')}"
    return text


def language_rule(language: str, mode: str) -> str:
    keep_text = (
        "Text that appears in the image must be written exactly as it is, "
        "never translated."
    )
    if mode == "text":
        return (
            f"\nIf you need to say something yourself (for example that there "
            f"is no text), say it in {language}. Never put any label or prefix "
            f"before the lines of text. {keep_text}"
        )
    if mode == "ask":
        return (
            f"\nAnswer in the same language the question is written in. If you "
            f"can't tell which language that is, answer in {language}. {keep_text}"
        )
    if language in LABELS:
        label_rule = (
            f"Wherever these rules say 'Text in image:', write exactly "
            f"'{LABELS[language]}:' instead, and write it only once."
        )
    else:
        label_rule = (
            f"Wherever these rules say 'Text in image:', translate that label "
            f"into {language}, always use the same wording, and write it only once."
        )
    return f"\nWrite your answer in {language}. {label_rule} {keep_text}"


def build_prompt(mode: str, language: str, animated: bool, question: str = "") -> str:
    rules = {"short": SHORT_RULES, "long": LONG_RULES, "text": TEXT_RULES, "ask": ASK_RULES}
    prompt = BASE_RULES + rules[mode]
    if mode == "ask":
        prompt += question.strip()
    if animated:
        prompt += ANIMATION_NOTE
    prompt += language_rule(language, mode)
    return prompt


def language_of(interaction: discord.Interaction | None) -> str:
    if interaction is None:
        return "English"
    return LANGUAGES.get(str(interaction.locale), "English")


# ---------- Setup ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pixelroom")

if not DISCORD_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("Missing DISCORD_TOKEN or GEMINI_API_KEY in the .env file.")

ai = genai.Client(api_key=GEMINI_API_KEY)

# We only need plain text answers, no tools
AI_CONFIG = types.GenerateContentConfig(
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)
limiter = asyncio.Semaphore(MAX_PARALLEL)

intents = discord.Intents.default()
intents.message_content = True  # needed to see attachments and GIF links
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# Messages already described in automatic mode
handled: OrderedDict[int, None] = OrderedDict()
HANDLED_MAX = 1000


def mark_handled(message_id: int) -> bool:
    """Remember a message. Returns False if it was already handled."""
    if message_id in handled:
        return False
    handled[message_id] = None
    if len(handled) > HANDLED_MAX:
        handled.popitem(last=False)
    return True


# ---------- Preparing files for the AI ----------

def clean_type(content_type: str | None) -> str:
    return (content_type or "").split(";")[0].strip().lower()


def is_supported(content_type: str) -> bool:
    return content_type in STILL_TYPES or content_type == GIF_TYPE or content_type in VIDEO_TYPES


def gif_to_frames(data: bytes) -> list[types.Part]:
    """Take a few evenly spaced frames from a GIF and turn them into PNGs."""
    with Image.open(io.BytesIO(data)) as gif:
        total = getattr(gif, "n_frames", 1)
        if total <= 1:
            wanted = [0]
        else:
            count = min(GIF_FRAMES, total)
            wanted = sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})

        parts = []
        for index, frame in enumerate(ImageSequence.Iterator(gif)):
            if index in wanted:
                buffer = io.BytesIO()
                frame.convert("RGB").save(buffer, format="PNG")
                parts.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"))
            if index >= wanted[-1]:
                break
    return parts


def to_parts(data: bytes, content_type: str) -> tuple[list[types.Part], bool]:
    """Turn downloaded bytes into AI input. Returns (parts, is_animated)."""
    if content_type == GIF_TYPE:
        frames = gif_to_frames(data)
        return frames, len(frames) > 1
    if content_type in VIDEO_TYPES:
        return [types.Part.from_bytes(data=data, mime_type=content_type)], True
    return [types.Part.from_bytes(data=data, mime_type=content_type)], False


def file_extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def needs_conversion(attachment: discord.Attachment) -> bool:
    content_type = clean_type(attachment.content_type)
    if content_type in VIDEO_TYPES:
        return False
    is_video = content_type.startswith("video/") or file_extension(attachment.filename) in CONVERT_EXTENSIONS
    return bool(FFMPEG) and is_video


def convert_to_mp4(data: bytes, extension: str) -> bytes:
    """Turn another video format into a small MP4 the AI can read (no sound)."""
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "in" + (extension or ".mkv"))
        target = os.path.join(folder, "out.mp4")
        with open(source, "wb") as f:
            f.write(data)
        attempts = [
            # Fast: just repackage the video without re-encoding it
            ["-c:v", "copy", "-an"],
            # Slower: re-encode, smaller picture, works for almost anything
            ["-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-an"],
        ]
        for options in attempts:
            command = [FFMPEG, "-y", "-loglevel", "error", "-i", source,
                       "-t", str(MAX_VIDEO_SECONDS), *options, "-movflags", "+faststart", target]
            result = subprocess.run(command, capture_output=True, timeout=90)
            if result.returncode == 0 and os.path.exists(target) and os.path.getsize(target) > 0:
                with open(target, "rb") as f:
                    return f.read()
        raise RuntimeError("ffmpeg could not convert the video: " + result.stderr.decode(errors="ignore")[-300:])


def unreadable_media(message: discord.Message) -> list[str]:
    """File names of attachments that look like images or videos but we can't read."""
    names = []
    for attachment in message.attachments:
        content_type = clean_type(attachment.content_type)
        looks_like_media = (
            content_type.startswith(("image/", "video/"))
            or file_extension(attachment.filename) in MEDIA_EXTENSIONS
        )
        readable = is_supported(content_type) or needs_conversion(attachment)
        if looks_like_media and (not readable or attachment.size > MAX_FILE_SIZE):
            names.append(attachment.filename)
    return names


# ---------- Finding images in a message ----------

def attachment_sources(
    message: discord.Message, skip_with_alt: bool = True
) -> list[tuple[str, discord.Attachment]]:
    """Image, GIF and video files attached to the message."""
    found = []
    for attachment in message.attachments:
        if not (is_supported(clean_type(attachment.content_type)) or needs_conversion(attachment)):
            continue
        if attachment.size > MAX_FILE_SIZE:
            log.info("Skipping %s: too large", attachment.filename)
            continue
        if skip_with_alt and attachment.description:  # the poster wrote alt text
            continue
        found.append((attachment.filename, attachment))
    return found


def link_preview_context(embed: discord.Embed) -> str:
    """Tell the AI that it only sees the preview picture of a linked video or page."""
    kind = "video" if embed.type == "video" or embed.video.url else "web page"
    details = []
    if embed.provider and embed.provider.name:
        details.append(f"site: {embed.provider.name}")
    if embed.author and embed.author.name:
        details.append(f"posted by: {embed.author.name}")
    if embed.title:
        details.append(f"title: {embed.title}")
    if embed.description:
        details.append(f"description: {embed.description[:300]}")
    info = "; ".join(details) if details else "no extra details"
    return (
        f"\nImportant: this is NOT the whole {kind}. It is only the preview "
        f"picture (thumbnail) of a linked {kind}. Link details ({info}). "
        f"Start by saying, in the answer language, that this is the preview of "
        f"a linked {kind} and from which site, and give its title or creator if "
        f"known. Then describe the preview picture. Do not guess what happens "
        f"in the {kind} beyond what the preview and the details show."
    )


def embed_sources(message: discord.Message) -> list[tuple[str, list[str], str]]:
    """Images, GIFs and link previews (Discord GIF picker, Tenor, image links,
    and the preview pictures of linked videos or pages like TikTok or YouTube).

    Each source is (label, urls to try in order, extra context for the AI).
    """
    found = []
    for embed in message.embeds:
        context = ""
        if embed.type == "gifv":
            urls = [
                embed.video.proxy_url, embed.video.url,
                embed.thumbnail.proxy_url, embed.thumbnail.url,
            ]
        elif embed.type == "image":
            urls = [
                embed.thumbnail.proxy_url, embed.thumbnail.url,
                embed.image.proxy_url, embed.image.url,
            ]
        elif embed.type in ("video", "rich", "link", "article"):
            # We can't download the video itself, only its preview picture
            urls = [
                embed.thumbnail.proxy_url, embed.thumbnail.url,
                embed.image.proxy_url, embed.image.url,
            ]
            context = link_preview_context(embed)
        else:
            continue
        urls = list(dict.fromkeys(u for u in urls if u))  # drop empty and duplicate
        if urls:
            found.append((f"{embed.type} link", urls, context))
    return found


def has_images(message: discord.Message) -> bool:
    return bool(attachment_sources(message, skip_with_alt=False) or embed_sources(message))


async def find_image_messages(channel, back: int, count: int):
    """The image messages in a channel, newest first, skipping (back - 1) of them.

    Also returns the newest file we had to skip because we can't read it,
    as (file name, poster), so we can tell the user instead of silently
    describing an older image.
    """
    found: list[discord.Message] = []
    skipped = None
    seen = 0
    async for message in channel.history(limit=HISTORY_LIMIT):
        if message.author == bot.user:
            continue
        if not found and skipped is None:
            bad = unreadable_media(message)
            if bad:
                skipped = (bad[0], message.author.display_name)
        if not has_images(message):
            continue
        seen += 1
        if seen >= back:
            found.append(message)
            if len(found) >= count:
                break
    return found, skipped


async def download_url(url: str) -> tuple[bytes, str]:
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            if response.content_length and response.content_length > MAX_FILE_SIZE:
                raise ValueError("file too large")
            data = await response.read()
            return data, clean_type(response.headers.get("Content-Type"))


# ---------- Talking to the AI ----------

async def ask_ai(
    label: str,
    parts: list[types.Part],
    animated: bool,
    mode: str,
    language: str,
    question: str = "",
    context: str = "",
) -> str | None:
    prompt = build_prompt(mode, language, animated, question) + context
    kind = parts[0].inline_data.mime_type if parts and parts[0].inline_data else "?"
    is_video = kind in VIDEO_TYPES
    log.info("Sending %s to %s (%s, mode=%s, lang=%s)", label, GEMINI_MODEL, kind, mode, language)
    timeout = AI_TIMEOUT_VIDEO if is_video else AI_TIMEOUT
    for attempt in range(1, AI_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            # The AI call runs in a separate thread so it can never block the bot
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ai.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=[prompt, *parts],
                    config=AI_CONFIG,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            log.warning(
                "AI did not answer within %d s for %s (attempt %d of %d)",
                timeout, label, attempt, AI_ATTEMPTS,
            )
            continue
        except Exception:
            log.exception("AI request failed for %s (attempt %d of %d)", label, attempt, AI_ATTEMPTS)
            if attempt < AI_ATTEMPTS:
                await asyncio.sleep(1)
            continue
        log.info(
            "AI answered for %s in %.1f s (attempt %d)",
            label, time.monotonic() - started, attempt,
        )
        text = (response.text or "").strip()
        if text:
            return text
    return None


async def describe_attachment(
    label: str, attachment: discord.Attachment, mode: str, language: str, question: str = ""
) -> str | None:
    async with limiter:
        try:
            log.info("Downloading %s (%d bytes)", label, attachment.size)
            data = await asyncio.wait_for(attachment.read(), timeout=30)
            content_type = clean_type(attachment.content_type)
            if needs_conversion(attachment):
                log.info("Converting %s to MP4", label)
                data = await asyncio.to_thread(
                    convert_to_mp4, data, file_extension(attachment.filename)
                )
                content_type = "video/mp4"
            parts, animated = await asyncio.to_thread(to_parts, data, content_type)
            return await ask_ai(label, parts, animated, mode, language, question)
        except Exception:
            log.exception("Could not describe %s", label)
            return None


async def describe_link(
    label: str, urls: list[str], mode: str, language: str, question: str = "",
    context: str = "",
) -> str | None:
    async with limiter:
        for url in urls:
            try:
                log.info("Downloading %s", label)
                data, content_type = await download_url(url)
                if not is_supported(content_type):
                    log.info("Skipping %s: unsupported type %s", label, content_type)
                    continue
                parts, animated = await asyncio.to_thread(to_parts, data, content_type)
                # ask_ai already retries; if it still fails, don't start over
                # with the next link, or the user would wait far too long
                # A still preview of a video is not an animation
                if context:
                    animated = False
                return await ask_ai(label, parts, animated, mode, language, question, context)
            except Exception:
                log.exception("Could not describe %s from one source, trying the next", label)
        return None


async def describe_sources(
    message: discord.Message,
    mode: str,
    language: str,
    question: str = "",
    skip_with_alt: bool = False,
) -> list[str]:
    files = attachment_sources(message, skip_with_alt=skip_with_alt)
    links = embed_sources(message)
    jobs = [describe_attachment(label, a, mode, language, question) for label, a in files]
    jobs += [
        describe_link(label, urls, mode, language, question, context)
        for label, urls, context in links
    ]
    results = [r for r in await asyncio.gather(*jobs) if r]
    if mode == "short":
        results = [trim_text_section(r, language) for r in results]
    return results


# ---------- Fixed texts in several languages ----------
# Languages not listed here fall back to English for these short texts.
# (The descriptions themselves always come in the user's language.)

TEXTS = {
    "English": {
        "skipped": "Note: the newest file, {name} from {who}, is in a format I can't read, so I'm describing the image before it.",
        "readtext_hint": 'For the full text, use /readtext.',
        "help": 'PixelRoom help. Every answer is private, only you can see it.\n\n/describe: describes the latest image, GIF or video in this channel.\nOptions: detail long gives a longer description. count 3 describes the last 3 images. back 2 describes the image before the latest one.\n\n/ask: ask a question about the latest image, for example: What colour is her dress?\n\n/readtext: reads out all the text in the latest image, exactly as written.\n\nDescribe image: for a specific message, open its menu, choose Apps, then Describe image. With a keyboard, focus the message and press Shift and F10.\n\nAnswers come in the language your Discord is set to.\nWebsite: https://pixelroombot.com\nSupport server: https://discord.gg/eYP9nK5bmz',
        "text_label": "Text in image",
        "question": "Question",
        "answer": "Answer",
        "posted": "Posted by {name}, {when}.",
        "now": "just now",
        "minutes": lambda n: f"{n} minute{'s' if n != 1 else ''} ago",
        "hours": lambda n: f"{n} hour{'s' if n != 1 else ''} ago",
        "days": lambda n: f"{n} day{'s' if n != 1 else ''} ago",
        "image": "Image",
        "sorry": "Sorry, I could not read this image right now. Please try again later.",
        "no_image": "I couldn't find an image, GIF or video in the recent messages here.",
        "no_image_msg": "This message has no image, GIF or video to describe.",
        "no_channel": "I can't read this channel.",
        "no_history": "I don't have permission to read the history of this channel.",
    },
    "Hungarian": {
        "skipped": 'Megjegyzés: a legújabb fájlt ({name}, {who}) nem tudom elolvasni, ezért az előtte lévő képet írom le.',
        "readtext_hint": 'A teljes szöveghez használd a /readtext parancsot.',
        "help": 'PixelRoom súgó. Minden válasz privát, csak te látod.\n\n/describe: leírja a csatorna legutóbbi képét, GIF-jét vagy videóját.\nBeállítások: a detail long hosszabb leírást ad. A count 3 az utolsó 3 képet írja le. A back 2 a legutóbbi előtti képet írja le.\n\n/ask: kérdezz a legutóbbi képről, például: Milyen színű a ruhája?\n\n/readtext: felolvassa a legutóbbi képen lévő összes szöveget, pontosan úgy, ahogy írva van.\n\nDescribe image: egy adott üzenethez nyisd meg az üzenet menüjét, válaszd az Apps, majd a Describe image lehetőséget. Billentyűzettel jelöld ki az üzenetet, és nyomd meg a Shift és F10 billentyűt.\n\nA válaszok azon a nyelven jönnek, amire a Discordod be van állítva.\nWeboldal: https://pixelroombot.com\nSupport szerver: https://discord.gg/eYP9nK5bmz',
        "text_label": "Szöveg a képen",
        "question": "Kérdés",
        "answer": "Válasz",
        "posted": "{name} posztolta, {when}.",
        "now": "épp most",
        "minutes": lambda n: f"{n} perce",
        "hours": lambda n: f"{n} órája",
        "days": lambda n: f"{n} napja",
        "image": "Kép",
        "sorry": "Sajnos most nem tudtam elolvasni ezt a képet. Kérlek, próbáld újra később.",
        "no_image": "Nem találtam képet, GIF-et vagy videót a legutóbbi üzenetek között.",
        "no_image_msg": "Ebben az üzenetben nincs leírható kép, GIF vagy videó.",
        "no_channel": "Ezt a csatornát nem tudom olvasni.",
        "no_history": "Nincs engedélyem a csatorna korábbi üzeneteinek olvasására.",
    },
    "German": {
        "skipped": 'Hinweis: Die neueste Datei, {name} von {who}, hat ein Format, das ich nicht lesen kann, deshalb beschreibe ich das Bild davor.',
        "readtext_hint": 'Für den ganzen Text nutze /readtext.',
        "help": 'PixelRoom Hilfe. Jede Antwort ist privat, nur du kannst sie sehen.\n\n/describe: beschreibt das neueste Bild, GIF oder Video in diesem Kanal.\nOptionen: detail long gibt eine längere Beschreibung. count 3 beschreibt die letzten 3 Bilder. back 2 beschreibt das Bild vor dem neuesten.\n\n/ask: stelle eine Frage zum neuesten Bild, zum Beispiel: Welche Farbe hat ihr Kleid?\n\n/readtext: liest den gesamten Text im neuesten Bild vor, genau so, wie er geschrieben ist.\n\nDescribe image: für eine bestimmte Nachricht öffne ihr Menü, wähle Apps und dann Describe image. Mit der Tastatur markiere die Nachricht und drücke Umschalt und F10.\n\nAntworten kommen in der Sprache, auf die dein Discord eingestellt ist.\nWebsite: https://pixelroombot.com\nSupport-Server: https://discord.gg/eYP9nK5bmz',
        "text_label": "Text im Bild",
        "question": "Frage",
        "answer": "Antwort",
        "posted": "Gepostet von {name}, {when}.",
        "now": "gerade eben",
        "minutes": lambda n: f"vor {n} Minute{'n' if n != 1 else ''}",
        "hours": lambda n: f"vor {n} Stunde{'n' if n != 1 else ''}",
        "days": lambda n: f"vor {n} Tag{'en' if n != 1 else ''}",
        "image": "Bild",
        "sorry": "Ich konnte dieses Bild gerade nicht lesen. Bitte versuche es später noch einmal.",
        "no_image": "Ich habe in den letzten Nachrichten kein Bild, GIF oder Video gefunden.",
        "no_image_msg": "Diese Nachricht enthält kein Bild, GIF oder Video.",
        "no_channel": "Ich kann diesen Kanal nicht lesen.",
        "no_history": "Ich darf den Verlauf dieses Kanals nicht lesen.",
    },
    "French": {
        "skipped": "Remarque : le fichier le plus récent, {name} de {who}, est dans un format que je ne peux pas lire, je décris donc l'image précédente.",
        "readtext_hint": 'Pour le texte complet, utilise /readtext.',
        "help": "Aide PixelRoom. Chaque réponse est privée, toi seul peux la voir.\n\n/describe : décrit la dernière image, GIF ou vidéo de ce salon.\nOptions : detail long donne une description plus longue. count 3 décrit les 3 dernières images. back 2 décrit l'image avant la dernière.\n\n/ask : pose une question sur la dernière image, par exemple : De quelle couleur est sa robe ?\n\n/readtext : lit tout le texte de la dernière image, exactement comme il est écrit.\n\nDescribe image : pour un message précis, ouvre son menu, choisis Apps, puis Describe image. Au clavier, sélectionne le message et appuie sur Maj et F10.\n\nLes réponses arrivent dans la langue de ton Discord.\nSite web : https://pixelroombot.com\nServeur d'aide : https://discord.gg/eYP9nK5bmz",
        "text_label": "Texte dans l'image",
        "question": "Question",
        "answer": "Réponse",
        "posted": "Publié par {name}, {when}.",
        "now": "à l'instant",
        "minutes": lambda n: f"il y a {n} minute{'s' if n != 1 else ''}",
        "hours": lambda n: f"il y a {n} heure{'s' if n != 1 else ''}",
        "days": lambda n: f"il y a {n} jour{'s' if n != 1 else ''}",
        "image": "Image",
        "sorry": "Désolé, je n'ai pas pu lire cette image. Réessaie plus tard.",
        "no_image": "Je n'ai trouvé aucune image, GIF ou vidéo dans les messages récents.",
        "no_image_msg": "Ce message ne contient pas d'image, de GIF ou de vidéo.",
        "no_channel": "Je ne peux pas lire ce salon.",
        "no_history": "Je n'ai pas la permission de lire l'historique de ce salon.",
    },
    "Spanish": {
        "skipped": 'Nota: el archivo más reciente, {name} de {who}, está en un formato que no puedo leer, así que describo la imagen anterior.',
        "readtext_hint": 'Para el texto completo, usa /readtext.',
        "help": 'Ayuda de PixelRoom. Cada respuesta es privada, solo tú puedes verla.\n\n/describe: describe la última imagen, GIF o vídeo de este canal.\nOpciones: detail long da una descripción más larga. count 3 describe las últimas 3 imágenes. back 2 describe la imagen anterior a la última.\n\n/ask: haz una pregunta sobre la última imagen, por ejemplo: ¿De qué color es su vestido?\n\n/readtext: lee todo el texto de la última imagen, exactamente como está escrito.\n\nDescribe image: para un mensaje concreto, abre su menú, elige Apps y luego Describe image. Con el teclado, selecciona el mensaje y pulsa Mayús y F10.\n\nLas respuestas llegan en el idioma de tu Discord.\nSitio web: https://pixelroombot.com\nServidor de ayuda: https://discord.gg/eYP9nK5bmz',
        "text_label": "Texto en la imagen",
        "question": "Pregunta",
        "answer": "Respuesta",
        "posted": "Publicado por {name}, {when}.",
        "now": "ahora mismo",
        "minutes": lambda n: f"hace {n} minuto{'s' if n != 1 else ''}",
        "hours": lambda n: f"hace {n} hora{'s' if n != 1 else ''}",
        "days": lambda n: f"hace {n} día{'s' if n != 1 else ''}",
        "image": "Imagen",
        "sorry": "Lo siento, no pude leer esta imagen ahora. Inténtalo más tarde.",
        "no_image": "No encontré ninguna imagen, GIF o vídeo en los mensajes recientes.",
        "no_image_msg": "Este mensaje no tiene ninguna imagen, GIF o vídeo.",
        "no_channel": "No puedo leer este canal.",
        "no_history": "No tengo permiso para leer el historial de este canal.",
    },
}


def tr(language: str, key: str):
    return TEXTS.get(language, TEXTS["English"])[key]


# ---------- Building the answer text ----------

def time_ago(moment, language: str = "English") -> str:
    """Plain words like 'just now', '5 minutes ago', '2 hours ago'."""
    seconds = int((discord.utils.utcnow() - moment).total_seconds())
    if seconds < 60:
        return tr(language, "now")
    minutes = seconds // 60
    if minutes < 60:
        return tr(language, "minutes")(minutes)
    hours = minutes // 60
    if hours < 24:
        return tr(language, "hours")(hours)
    return tr(language, "days")(hours // 24)


def join_results(results: list[str], language: str = "English") -> str:
    if len(results) == 1:
        return results[0]
    word = tr(language, "image")
    return "\n\n".join(f"{word} {i}: {text}" for i, text in enumerate(results, start=1))


def header(message: discord.Message, language: str = "English") -> str:
    return tr(language, "posted").format(
        name=message.author.display_name, when=time_ago(message.created_at, language)
    )


def split_message(text: str) -> list[str]:
    """Split long text into Discord-sized pieces, preferring paragraph breaks."""
    chunks = []
    while len(text) > DISCORD_LIMIT:
        cut = text.rfind("\n\n", 0, DISCORD_LIMIT)
        if cut < DISCORD_LIMIT // 2:
            cut = text.rfind(" ", 0, DISCORD_LIMIT)
        if cut <= 0:
            cut = DISCORD_LIMIT
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


async def send_private(interaction: discord.Interaction, text: str) -> None:
    for chunk in split_message(text):
        await interaction.followup.send(chunk, ephemeral=True)




async def get_targets(interaction: discord.Interaction, back: int, count: int):
    """Find image messages for a command, or tell the user why not."""
    language = language_of(interaction)
    channel = interaction.channel
    if channel is None or not hasattr(channel, "history"):
        await interaction.followup.send(tr(language, "no_channel"), ephemeral=True)
        return None
    try:
        targets, skipped = await find_image_messages(channel, back, count)
    except discord.Forbidden:
        await interaction.followup.send(
            tr(language, "no_history"), ephemeral=True
        )
        return None
    note = tr(language, "skipped").format(name=skipped[0], who=skipped[1]) if skipped else ""
    if not targets:
        message = f"{note}\n{tr(language, 'no_image')}" if note else tr(language, "no_image")
        await interaction.followup.send(message, ephemeral=True)
        return None
    return targets, note


# ---------- Commands ----------

@tree.command(
    name="describe",
    description="Describe the latest image, GIF or video in this channel (only you will see it)",
)
@app_commands.describe(
    back="1 = the latest image (default), 2 = the one before it, and so on",
    count="How many images to describe, starting from 'back' (default 1)",
    detail="Short (default) or long, detailed description",
)
@app_commands.choices(
    detail=[
        app_commands.Choice(name="short", value="short"),
        app_commands.Choice(name="long", value="long"),
    ]
)
async def describe_command(
    interaction: discord.Interaction,
    back: app_commands.Range[int, 1, 10] = 1,
    count: app_commands.Range[int, 1, 5] = 1,
    detail: app_commands.Choice[str] | None = None,
):
    await interaction.response.defer(ephemeral=True, thinking=True)
    mode = detail.value if detail else "short"
    language = language_of(interaction)
    log.info("/describe back=%d count=%d detail=%s by %s", back, count, mode, interaction.user)

    found = await get_targets(interaction, back, count)
    if not found:
        return
    targets, note = found

    # Oldest first, so the answer reads in the order things were posted
    targets.reverse()
    results = await asyncio.gather(*(describe_sources(m, mode, language) for m in targets))

    blocks = []
    for message, descriptions in zip(targets, results):
        body = join_results(descriptions, language) if descriptions else tr(language, "sorry")
        blocks.append(f"{header(message, language)} {body}")
    if note:
        blocks.insert(0, note)
    await send_private(interaction, "\n\n".join(blocks))


@tree.command(name="ask", description="Ask a question about an image in this channel (only you will see it)")
@app_commands.describe(
    question="Your question, for example: What colour is her dress?",
    back="1 = the latest image (default), 2 = the one before it, and so on",
)
async def ask_command(
    interaction: discord.Interaction,
    question: app_commands.Range[str, 1, 300],
    back: app_commands.Range[int, 1, 10] = 1,
):
    await interaction.response.defer(ephemeral=True, thinking=True)
    language = language_of(interaction)
    log.info("/ask back=%d by %s", back, interaction.user)

    found = await get_targets(interaction, back, 1)
    if not found:
        return
    targets, note = found
    prefix = f"{note}\n\n" if note else ""

    answers = await describe_sources(targets[0], "ask", language, question)
    if answers:
        body = (
            f"{tr(language, 'question')}: {question.strip()} "
            f"{tr(language, 'answer')}: {join_results(answers, language)}"
        )
    else:
        body = tr(language, "sorry")
    await send_private(interaction, f"{prefix}{header(targets[0], language)} {body}")


@tree.command(name="readtext", description="Read out all the text in an image in this channel (only you will see it)")
@app_commands.describe(back="1 = the latest image (default), 2 = the one before it, and so on")
async def readtext_command(
    interaction: discord.Interaction,
    back: app_commands.Range[int, 1, 10] = 1,
):
    await interaction.response.defer(ephemeral=True, thinking=True)
    language = language_of(interaction)
    log.info("/readtext back=%d by %s", back, interaction.user)

    found = await get_targets(interaction, back, 1)
    if not found:
        return
    targets, note = found
    prefix = f"{note}\n\n" if note else ""

    texts = [strip_labels(t) for t in await describe_sources(targets[0], "text", language)]
    body = join_results(texts, language) if texts else tr(language, "sorry")
    await send_private(interaction, f"{prefix}{header(targets[0], language)}\n{body}")


@tree.context_menu(name="Describe image")
async def describe_image_menu(interaction: discord.Interaction, message: discord.Message):
    await interaction.response.defer(ephemeral=True, thinking=True)
    language = language_of(interaction)
    if not has_images(message):
        await interaction.followup.send(tr(language, "no_image_msg"), ephemeral=True)
        return
    log.info("Describe image (menu) by %s", interaction.user)
    descriptions = await describe_sources(message, "short", language)
    body = join_results(descriptions, language) if descriptions else tr(language, "sorry")
    await send_private(interaction, f"{header(message, language)} {body}")


# ---------- Automatic mode (off by default) ----------

async def describe_message_publicly(message: discord.Message) -> None:
    if not attachment_sources(message) and not embed_sources(message):
        return
    if not mark_handled(message.id):
        return
    async with message.channel.typing():
        descriptions = await describe_sources(message, "short", "English", skip_with_alt=True)
    if not descriptions:
        return
    try:
        for chunk in split_message("Image description: " + join_results(descriptions)):
            await message.reply(
                chunk, mention_author=False, allowed_mentions=discord.AllowedMentions.none()
            )
    except discord.HTTPException:
        log.exception("Could not send the description")


@tree.command(name="help", description="How to use PixelRoom (only you will see it)")
async def help_command(interaction: discord.Interaction):
    log.info("/help by %s", interaction.user)
    await interaction.response.send_message(
        tr(language_of(interaction), "help"),
        ephemeral=True,
        suppress_embeds=True,  # no link previews, so the screen reader stays quiet
    )


# ---------- Safety net: never leave a command "thinking" forever ----------

@tree.error
async def on_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    log.exception("Command failed for %s", interaction.user, exc_info=error)
    message = tr(language_of(interaction), "sorry")
    try:
        if interaction.response.is_done():
            await interaction.followup.send(message, ephemeral=True)
        else:
            await interaction.response.send_message(message, ephemeral=True)
    except discord.HTTPException:
        log.exception("Could not send the error message")


# ---------- Events ----------

@bot.event
async def setup_hook():
    # Register the commands with Discord
    synced = await tree.sync()
    log.info("Registered %d command(s)", len(synced))


@bot.event
async def on_ready():
    mode = "automatic + on request" if AUTO_DESCRIBE else "on request only"
    log.info("Logged in as %s (servers: %d, mode: %s)", bot.user, len(bot.guilds), mode)


@bot.event
async def on_message(message: discord.Message):
    if not AUTO_DESCRIBE or message.author.bot:
        return
    await describe_message_publicly(message)


@bot.event
async def on_message_edit(before: discord.Message, after: discord.Message):
    # Discord often adds the GIF preview a moment after the message is sent
    if not AUTO_DESCRIBE or after.author.bot:
        return
    if after.embeds and not before.embeds:
        await describe_message_publicly(after)


bot.run(DISCORD_TOKEN, log_handler=None)
