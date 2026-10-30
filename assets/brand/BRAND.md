# Olds College brand assets used by OCCD

Source: www.oldscollege.ca (public site header/footer images and stylesheet, 2026-10-26). Replace with the official
brand files from Olds College Marketing when available (vector logo, brand guide).

Logos
- oc_logo_horizontal.png           500 x 125, RGB, white background   -> light sidebar / header
- oc_logo_horizontal_reversed.png  1967 x 495, RGBA, white wordmark   -> dark backgrounds
- oc_logo_shield.png               320 x 244, RGBA, shield + wordmark  -> favicon / title page

Palette (dominant colours of the logo, confirmed in the site stylesheet)
- Navy blue    #00358E  primary (wordmark)         -> primaryColor, headings, primary buttons
- Light blue   #009CDE  (shield)                   -> links, accents, selected states
- Green        #00B373  (shield)                   -> success / kept
- Dark green   #006747  (shield)                   -> secondary accent
- Orange       #F5A623  (hexagon, approximate)     -> warnings / removed-point highlight
- Site greys   #2F292B text, #F8F5F5 background, #DFD4D5 borders

Typography on the site: Open Sans (body), Julius Sans One (display). The app uses the system sans-serif (theme
font = "sans-serif": no external font or CDN, so it runs offline on a field laptop).

Where the palette lives in the code: app/brand.py (NAME, VERSION, the hex constants, the logo paths, page_icon(),
inject() with the page's small CSS, map_colors for pydeck), .streamlit/config.toml (the Streamlit theme; the same
flags in the "occd" launch configuration), engine/clean/yield_clean.py (figure colours: navy lines, green kept,
orange removed, light-blue field outline, RdYlGn yield ramp) and engine/report/clean_report.py (navy headings and
table headers, #EEF4FB zebra rows, #C9D8EC rules, the horizontal logo in the masthead, the brand in the footer).
Text in light blue, green or orange on white fails WCAG AA (3.1:1, 2.7:1, 2.0:1): use the *_TEXT shades of
app/brand.py (#0077B6, #00784D, #9A5B00) when a word must carry the colour.
