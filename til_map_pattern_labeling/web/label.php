<html>
<head>
<title>label</title>
</head>
<body>

<?PHP
$page = $_GET['page'];
$im_id = $_GET['im_id'];
$userid = $_GET['userid'];

$savef = sprintf("./label_files/im_id_label%d_%d.txt", $userid, $im_id);
if (isset($_POST['cr']))
    unlink($savef);
else {
    if ($myfile = fopen($savef, 'r'))
        $class = fread($myfile, 20);
    else
        $class = '';

    $myfile = fopen($savef, 'w') or die('Unable to open file for status saving! Please contact Le Hou');
    if      (isset($_POST['c1']))
        fwrite($myfile, sprintf("c1"));
    else if (isset($_POST['c2']))
        fwrite($myfile, sprintf("c2"));
    else if (isset($_POST['c3']))
        fwrite($myfile, sprintf("c3"));
    else if (isset($_POST['c4']))
        fwrite($myfile, sprintf("c4"));
    else if (isset($_POST['c5']))
        fwrite($myfile, sprintf("c5"));
    else if (isset($_POST['c6']))
        fwrite($myfile, sprintf("c6"));
    else if (isset($_POST['c7']))
        fwrite($myfile, sprintf("c7"));
    else if (strpos($class,'c1')!==false)
        fwrite($myfile, sprintf("c1"));
    else if (strpos($class,'c2')!==false)
        fwrite($myfile, sprintf("c2"));
    else if (strpos($class,'c3')!==false)
        fwrite($myfile, sprintf("c3"));
    else if (strpos($class,'c4')!==false)
        fwrite($myfile, sprintf("c4"));
    else if (strpos($class,'c5')!==false)
        fwrite($myfile, sprintf("c5"));
    else if (strpos($class,'c6')!==false)
        fwrite($myfile, sprintf("c6"));
    else if (strpos($class,'c7')!==false)
        fwrite($myfile, sprintf("c7"));

    fclose($myfile);
}

$url = sprintf("Location: view_patches.php?page=%d", $page);
header($url);
?>

</body>
</html>
