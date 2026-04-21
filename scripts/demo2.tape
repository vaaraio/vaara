# Vaara regulator demo — renders docs/demo2.gif
# Build: vhs scripts/demo2.tape

Output docs/demo2.gif

Set Shell "bash"
Set FontSize 22
Set Width 1200
Set Height 720
Set Framerate 24
Set TypingSpeed 55ms
Set Padding 60

Set Theme { "name": "vaara-site", "background": "#0b0d11", "foreground": "#c8e6d4", "cursor": "#c8e6d4", "black": "#0b0d11", "white": "#c8e6d4", "brightBlack": "#6b7a6f", "brightWhite": "#e8f4ec", "red": "#c07070", "green": "#6aaf88", "yellow": "#c8b56a", "blue": "#78a08a", "magenta": "#b08aa0", "cyan": "#78a08a", "brightRed": "#d48585", "brightGreen": "#7fd5a5", "brightYellow": "#d4c380", "brightBlue": "#8fb8a1", "brightMagenta": "#bf9aae", "brightCyan": "#8fb8a1" }

Hide
Type "export PS1='$ ' && clear"
Enter
Show

Sleep 500ms
Type "python examples/governance_demo.py"
Enter
Sleep 7500ms
