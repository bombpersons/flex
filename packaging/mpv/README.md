# Flex's patches to mpv

Flex can use any libmpv, but these two patches let it do things an ordinary one can't:

- `0001`: draws subtitles into a framebuffer of their own, so video effects leave them alone
  and they can be laid on top.
- `0002`: a `flextap` audio filter and `mpv_flex_set_audio_tap`, which hand the samples being
  played to Flex. Effects now record the computer's own sound instead (`system_sound.rs` in
  flex-app), and only use this where that can't be done.

`v0.40.0/` is built on Debian 13 for the Linux package. `windows/` is the same change against the
commit in `windows/COMMIT`, which needs newer libraries than Debian has. It isn't built yet, and the
Windows package ships an ordinary libmpv. Flex looks for `mpv_flex_set_audio_tap` when it loads libmpv. If it
isn't there, subtitles go through the effect and sound comes from a filter inside mpv, as before.

    git clone -b v0.40.0 https://github.com/mpv-player/mpv
    cd mpv
    git am /path/to/flex/packaging/mpv/v0.40.0/*.patch

`.github/scripts/build-libmpv-linux.sh` builds them for release. `build-libmpv-windows.sh` is for the
Windows patch, and isn't part of the build.
