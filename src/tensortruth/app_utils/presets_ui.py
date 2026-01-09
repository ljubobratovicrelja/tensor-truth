"""Preset management UI components."""

import streamlit as st

from tensortruth.app_utils import apply_preset, quick_launch_preset, toggle_favorite


def render_favorite_preset_cards(
    favorites: dict, available_mods: list, presets_file: str, sessions_file: str
):
    """Render favorite presets as clickable cards in a grid.

    Args:
        favorites: Dict of favorite preset configs
        available_mods: List of available module names
        presets_file: Path to presets JSON file
        sessions_file: Path to sessions JSON file
    """
    if not favorites:
        return

    st.caption("One-click start with your favorite configurations")

    # Display favorites in cards (3 per row)
    fav_items = list(favorites.items())
    num_cols = 3
    num_rows = (len(fav_items) + num_cols - 1) // num_cols

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            item_idx = row * num_cols + col_idx
            if item_idx < len(fav_items):
                preset_name, preset_config = fav_items[item_idx]
                with cols[col_idx]:
                    with st.container(border=True, height=200):
                        st.markdown(f"**{preset_name}**")

                        # Show description if available
                        description = preset_config.get("description", "")
                        if description:
                            st.caption(description)
                        else:
                            # Fallback to module count and device
                            num_modules = len(preset_config.get("modules", []))
                            device = preset_config.get("llm_device", "gpu").upper()
                            st.caption(f"{num_modules} modules • {device}")

                        if st.button(
                            "LAUNCH",
                            key=f"launch_{preset_name}",
                            type="primary",
                            use_container_width=True,
                        ):
                            success, error = quick_launch_preset(
                                preset_name,
                                available_mods,
                                presets_file,
                                sessions_file,
                            )
                            if success:
                                st.session_state.mode = "chat"
                                st.rerun()
                            else:
                                st.error(error)

    st.markdown("---")


def render_presets_manager(
    presets: dict,
    available_mods: list,
    available_models: list,
    system_devices: list,
    presets_file: str,
):
    """Render the presets manager expander with load/delete/favorite actions.

    Args:
        presets: Dict of all presets
        available_mods: List of available module names
        available_models: List of available Ollama models
        system_devices: List of system devices (cpu, cuda, mps)
        presets_file: Path to presets JSON file
    """
    if not presets:
        return

    # Collapse presets when config section is expanded
    expand_presets = not st.session_state.get("expand_config_section", False)

    with st.expander("Presets", expanded=expand_presets):
        for preset_name in presets.keys():
            is_favorite = presets[preset_name].get("favorite", False)
            star_icon = "⭐" if is_favorite else "☆"

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"**{preset_name}**")
            with col2:
                if st.button(
                    "Load", key=f"load_{preset_name}", use_container_width=True
                ):
                    apply_preset(
                        preset_name,
                        available_mods,
                        available_models,
                        system_devices,
                        presets_file,
                    )
                    # Expand the config section to show loaded values
                    st.session_state.expand_config_section = True
                    st.rerun()
            with col3:
                if st.button(
                    "Delete", key=f"del_{preset_name}", use_container_width=True
                ):
                    st.session_state.show_preset_delete_confirm = True
                    st.session_state.preset_to_delete = preset_name
                    st.rerun()
            with col4:
                if st.button(
                    star_icon,
                    key=f"fav_{preset_name}",
                    use_container_width=True,
                ):
                    toggle_favorite(preset_name, presets_file)
                    st.rerun()
