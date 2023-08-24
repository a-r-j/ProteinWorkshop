BUILDDIR=build  # Adapt to the location of your Sphinx output
for i in "$BUILDDIR"/*.html; do
    python emojize.py "$i" > "$i".new
    # or emojize_pngorsvg.py "$i" > "$i".new
    wdiff -3 "$i" "$i".new;
    mv -vf "$i".new "$i"
done