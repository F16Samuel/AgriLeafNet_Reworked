$SourceDir = "PlantVillage"
$DestDir = "dataset"

$TrainDir = Join-Path $DestDir "train"
$ValDir = Join-Path $DestDir "val"
$TestDir = Join-Path $DestDir "test"

# Create destination directories
New-Item -ItemType Directory -Force -Path $TrainDir, $ValDir, $TestDir | Out-Null

# Get all class folders
$ClassDirs = Get-ChildItem -Path $SourceDir -Directory

foreach ($Class in $ClassDirs) {
    $ClassName = $Class.Name
    $ClassPath = $Class.FullName

    # Create class subdirs
    $TrainClassDir = Join-Path $TrainDir $ClassName
    $ValClassDir = Join-Path $ValDir $ClassName
    $TestClassDir = Join-Path $TestDir $ClassName

    New-Item -ItemType Directory -Force -Path $TrainClassDir, $ValClassDir, $TestClassDir | Out-Null

    # Get image files (you can filter by extension if needed)
    $ImageFiles = Get-ChildItem -Path $ClassPath -File | Get-Random -Count ([int]::MaxValue)

    if ($ImageFiles.Count -lt 6) {
        Write-Warning "Not enough images in $ClassName (found $($ImageFiles.Count))"
        continue
    }

    # Copy 5 to TEST
    $TestImages = $ImageFiles[0..4]
    foreach ($img in $TestImages) {
        Copy-Item $img.FullName -Destination $TestClassDir
    }

    # Remaining images
    $RemainingImages = $ImageFiles[5..($ImageFiles.Count - 1)]
    $TrainCount = [math]::Floor($RemainingImages.Count * 0.8)

    $TrainImages = $RemainingImages[0..($TrainCount - 1)]
    $ValImages = $RemainingImages[$TrainCount..($RemainingImages.Count - 1)]

    foreach ($img in $TrainImages) {
        Copy-Item $img.FullName -Destination $TrainClassDir
    }

    foreach ($img in $ValImages) {
        Copy-Item $img.FullName -Destination $ValClassDir
    }

    Write-Host "✅ $ClassName - Total: $($ImageFiles.Count), Test: 5, Train: $($TrainImages.Count), Val: $($ValImages.Count)"
}

Write-Host "`n🎉 Dataset split completed."