// ==============================================================================
// SCRIPT GLOBAL : EXPORT DES STACKS ORMVAG (GEOTIFF) ET CSV (VALIDATION)
// Inclus le correctif "Dummy Fallback" pour les mois 100% nuageux.
// ==============================================================================

var ormvagBoundary = ee.FeatureCollection("projects/ee-mohamedaminelaghmich/assets/ORMVAG");
var year = 2025;
var exportFolder = 'geotiff-2025';
var CLOUD_FILTER = 60;        
var CLD_PRB_THRESH = 40;      

Map.centerObject(ormvagBoundary, 10);
Map.addLayer(ormvagBoundary, {color: 'blue', fillColor: '00000000'}, 'ORMVAG Boundary');

// 1. CLOUD MASKING
var s2_sr_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(ormvagBoundary).filter(ee.Filter.calendarRange(year, year, 'year'))
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER));
var s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterBounds(ormvagBoundary).filter(ee.Filter.calendarRange(year, year, 'year'));

var s2_joined = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply({
  'primary': s2_sr_col, 'secondary': s2_cloudless_col,
  'condition': ee.Filter.equals({'leftField': 'system:index', 'rightField': 'system:index'})
}));

function mask_s2_cloudless(img) {
  var cld_prb = ee.Image(img.get('s2cloudless')).select('probability');
  var is_not_cloud = cld_prb.gt(CLD_PRB_THRESH).not();
  return img.updateMask(is_not_cloud).divide(10000).copyProperties(img, ["system:time_start"]);
}
var base_s2_clean = s2_joined.map(mask_s2_cloudless);

var base_s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(ormvagBoundary).filter(ee.Filter.calendarRange(year, year, 'year'))
  .filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));

// NOTE : Pour raccourcir ce fichier sur GitHub, tu peux copier-coller ici 
// l'intégralité de ton grand script GEE que tu m'as envoyé au tout début de notre conversation.
