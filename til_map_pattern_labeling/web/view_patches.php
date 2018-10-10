<html>
<head>
<?php $y = $_COOKIE["y"];?>
<title>TIL-labeling</title>
<style>
.page_container {
    width: 100%;
    margin: auto;
}
.instance_container {
    float: left;
    padding: 2px;
    border: 2px solid #000000;
}
.info_frame {
    float: top;
    padding: 10px;
    background-color: #808080;
}
.label_frame {
    float: left;
    padding: 5px;
    border: 2px solid #808080;
}
.img_frame {
    float: top;
}
</style>
</head>
<body>

<?php
print "<body onScroll=\"document.cookie='y=' + window.pageYOffset\" onLoad='window.scrollTo(0,$y)'>";
?>

<?PHP
$page = $_GET['page'];
?>

<a href=view_patches.php?page=<?PHP printf("%d", $page - 1);?>>Prev Page</a>
<a href=view_patches.php?page=<?PHP printf("%d", $page + 1);?>>Next Page</a>
<p>
<?PHP printf('You are in page [<b>%d</b>]', $page); ?>
<p>

<?PHP
$N_per_page = 2;
$folder = 'images';

printf("<section class=\"page_container\">\n\n");
for ($i = 1; $i <= $N_per_page; ++$i) {
    $im_id = $i + $page * $N_per_page;

    $savef = sprintf("./label_files/im_id_label0_%d.txt", $im_id);
    if ($myfile = fopen($savef, 'r'))
        $class0 = fread($myfile, 20);
    else
        $class0 = '';

    $savef = sprintf("./label_files/im_id_label1_%d.txt", $im_id);
    if ($myfile = fopen($savef, 'r'))
        $class1 = fread($myfile, 20);
    else
        $class1 = '';

    printf("<section class=\"instance_container\">\n");

    printf("<div class=\"info_frame\">\n");
    $info_file = sprintf("%s/%d_info.txt", $folder, $im_id);
    if ($myfile = fopen($info_file, 'r')) {
        $svs_id = fread($myfile, 20);
        printf("Image: %s\n", $svs_id);
    }
    printf("</div>\n");

    printf("<div class=\"label_frame\">\n");
    printf("User-0\n");

    printf("<form name=\"label\" method=\"POST\" action=label.php?page=%d&im_id=%d&userid=%d>\n", $page, $im_id, 0);
    printf("<input type=\"submit\" name=\"c1\" value=\"%s\"> None <br>\n", strpos($class0,'c1')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c2\" value=\"%s\"> Non-Brisk, Focal <br>\n", strpos($class0,'c2')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c3\" value=\"%s\"> Non-Brisk, Multifocal <br>\n", strpos($class0,'c3')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c4\" value=\"%s\"> Brisk, Band-like <br>\n", strpos($class0,'c4')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c5\" value=\"%s\"> Brisk, Diffuse <br>\n", strpos($class0,'c5')!==false ? 'x':'  ');
    printf("<br>\n");
    printf("<input type=\"submit\" name=\"c6\" value=\"%s\"> Borderline <br>\n", strpos($class0,'c6')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c7\" value=\"%s\"> Indeterminate <br>\n", strpos($class0,'c7')!==false ? 'x':'  ');
    printf("<br>\n");
    printf("<input type=\"submit\" name=\"cr\" value=\"  \"> Clear Label\n");
    printf("</form>\n");
    printf("</div>\n");

    printf("<div class=\"label_frame\">\n");
    printf("User-1\n");

    printf("<form name=\"label\" method=\"POST\" action=label.php?page=%d&im_id=%d&userid=%d>\n", $page, $im_id, 1);
    printf("<input type=\"submit\" name=\"c1\" value=\"%s\"> None <br>\n", strpos($class1,'c1')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c2\" value=\"%s\"> Non-Brisk, Focal <br>\n", strpos($class1,'c2')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c3\" value=\"%s\"> Non-Brisk, Multifocal <br>\n", strpos($class1,'c3')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c4\" value=\"%s\"> Brisk, Band-like <br>\n", strpos($class1,'c4')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c5\" value=\"%s\"> Brisk, Diffuse <br>\n", strpos($class1,'c5')!==false ? 'x':'  ');
    printf("<br>\n");
    printf("<input type=\"submit\" name=\"c6\" value=\"%s\"> Borderline <br>\n", strpos($class1,'c6')!==false ? 'x':'  ');
    printf("<input type=\"submit\" name=\"c7\" value=\"%s\"> Indeterminate <br>\n", strpos($class1,'c7')!==false ? 'x':'  ');
    printf("<br>\n");
    printf("<input type=\"submit\" name=\"cr\" value=\"  \"> Clear Label\n");
    printf("</form>\n");
    printf("</div>\n");

    printf("<div class=\"img_frame\">\n");
    printf("<img height=250px src=\"%s/%d_HE.png\"/>\n", $folder, $im_id);
    printf("</div>\n");
    printf("<div class=\"img_frame\">\n");
    printf("<img height=250px src=\"%s/%d_TIL.png\"/>\n", $folder, $im_id);
    printf("</div>\n");
    printf("<div class=\"img_frame\">\n");
    printf("<img height=250px src=\"%s/%d_cluster.jpg\"/>\n", $folder, $im_id);
    printf("</div>\n");
    printf("</section>\n\n");
}
printf("</section>\n");
?>

</body>
</html>
