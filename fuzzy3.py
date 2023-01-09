def get_z(results):
    z_values = {
        ["BL", "BL", "BL", "BL", "BL"]: "BL",
        ["BW", "SL", "BL", "SL", "BW"]: "BL",
        ["SW", "BL", "SL", "SL", "SW"]: "BL",
        ["SL", "SL", "SL", "SL", "SL"]: "SL",
        ["D", "SL", "SL", "D", "D"]: "SL",
        ["SW", "D", "SL", "SL", "SW"]: "SL",
        ["D", "D", "D", "D", "D"]: "D",
        ["SL", "SW", "SW", "D", "SL"]: "D",
        ["D", "D", "SW", "SW", "D"]: "D",
        ["SW", "SW", "SW", "SW", "SW"]: "SW",
        ["D", "BW", "BW", "SW", "D"]: "SW",
        ["SL", "SW", "SW", "BW", "SL"]: "SW",
        ["BW", "BW", "BW", "BW", "BW"]: "BW",
        ["SL", "BW", "SW", "BW", "SL"]: "BW",
        ["BL", "SW", "BW", "SW", "BL"]: "BW",
    }
    return z_values.get(results, "UNKNOWN")
