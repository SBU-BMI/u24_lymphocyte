BEGIN {
    FS = "\"";
}

{
    gsub(":", "");
    gsub("{", "");
    gsub("}", "");
    gsub("\\[", "");
    gsub("\\]", "");
    for (i=1; i<=NF; ++i) {
        if ($i == "case_id")
            case_id = $(i+2);
        else if ($i == "username")
            username = $(i+2);
        else if ($i == "mark_type")
            mark_type = $(i+2);
        else if ($i == "mark_width") {
            mark_width = $(i+1);
            sub(",", "", mark_width);
        }
        else if ($i == "date") {
            date = $(i+1);
            sub(",", "", date);
        }
        else if (match($i, "coordinates"))
            coordinates = $(i+1);
    }
    print case_id"\t"username"\t"mark_type"\t"mark_width"\t"date"\t"coordinates;
}

