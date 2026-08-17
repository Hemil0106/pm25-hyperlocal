import argparse
from pathlib import Path

from src.config import load_config, get_project_root
from src.utils import (
    setup_logging,
    ensure_directories,
    validate_config,
)
from src.geospatial.aoi import load_aoi
from src.data_pipeline import (
    run_aod_pipeline,
    run_cpcb_pipeline,
    run_weather_pipeline,
    run_ndvi_pipeline,
    run_dem_pipeline,
    run_osm_pipeline,
    run_viirs_pipeline,
    run_alignment_pipeline,
    run_training_data_pipeline,
    run_model_pipeline,
    run_inference_pipeline,
    run_downscaling_pipeline,
    run_final_outputs_pipeline,
    create_cpcb_sample_data,
    create_sample_weather_data,
    create_sample_ndvi_data,
    create_sample_dem_data,
    create_sample_osm_data,
    create_sample_viirs_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="PM2.5 Hyperlocal Mapping pipeline"
    )
    parser.add_argument(
        "--create-sample-cpcb",
        action="store_true",
        help="Generate synthetic CPCB sample data and exit.",
    )
    parser.add_argument(
        "--cpcb-only",
        action="store_true",
        help="Run only the CPCB ingestion and preprocessing stage.",
    )
    parser.add_argument(
        "--create-sample-weather",
        action="store_true",
        help="Generate synthetic weather NetCDF sample data and exit.",
    )
    parser.add_argument(
        "--weather-only",
        action="store_true",
        help="Run only the weather ingestion and preprocessing stage.",
    )
    parser.add_argument(
        "--create-sample-ndvi",
        action="store_true",
        help="Generate synthetic MODIS NDVI sample GeoTIFF and exit.",
    )
    parser.add_argument(
        "--ndvi-only",
        action="store_true",
        help="Run only the NDVI ingestion and preprocessing stage.",
    )
    parser.add_argument(
        "--create-sample-dem",
        action="store_true",
        help="Generate synthetic SRTM elevation sample GeoTIFF and exit.",
    )
    parser.add_argument(
        "--dem-only",
        action="store_true",
        help="Run only the DEM/elevation ingestion and preprocessing stage.",
    )
    parser.add_argument(
        "--create-sample-osm",
        action="store_true",
        help="Generate synthetic OSM road-network GeoJSON sample and exit.",
    )
    parser.add_argument(
        "--osm-only",
        action="store_true",
        help="Run only the OSM road-density ingestion stage.",
    )
    parser.add_argument(
        "--create-sample-viirs",
        action="store_true",
        help="Generate synthetic VIIRS night-light sample GeoTIFF and exit.",
    )
    parser.add_argument(
        "--viirs-only",
        action="store_true",
        help="Run only the VIIRS night-time-lights ingestion stage.",
    )
    parser.add_argument(
        "--alignment-only",
        action="store_true",
        help="Run only the master data alignment stage using existing processed outputs.",
    )
    parser.add_argument(
        "--training-data-only",
        action="store_true",
        help="Run only the training-dataset + feature-engineering stage using existing alignment outputs.",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Run only the model-training + spatial CV stage using the existing training dataset.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run only the 1 km PM2.5 spatial inference + raster stage using existing model artifacts.",
    )
    parser.add_argument(
        "--downscaling-only",
        action="store_true",
        help="Run only the 500 m spatial downscaling / refinement stage using existing 1 km predictions.",
    )
    parser.add_argument(
        "--aqi-hotspot-only",
        action="store_true",
        help="Run only the AQI + hotspot + uncertainty + final geospatial outputs stage using existing 500 m predictions.",
    )
    parser.add_argument(
        "--global-data-only",
        action="store_true",
        help="Run only the Milestone 16 global data acquisition pipeline for the configured global scope.",
    )
    parser.add_argument(
        "--india-data-only",
        action="store_true",
        help="Run only the Milestone 16 global data acquisition pipeline for the India scope.",
    )
    parser.add_argument(
        "--delhi-data-only",
        action="store_true",
        help="Run only the Milestone 16 global data acquisition pipeline for the Delhi scope.",
    )
    parser.add_argument(
        "--global-training-only",
        action="store_true",
        help="Run only the Milestone 17 global training dataset + feature engineering "
             "stage from existing real Milestone 16 outputs.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date for inference/downscaling/outputs in YYYY-MM-DD format (default from config).",
    )
    args = parser.parse_args()

    project_root = get_project_root()

    config = load_config()

    logger = setup_logging(
        project_root / config["paths"]["logs"]
    )

    logger.info("=" * 70)
    logger.info("PM2.5 HYPERLOCAL MAPPING PIPELINE")
    logger.info("=" * 70)

    if args.create_sample_cpcb:
        create_cpcb_sample_data(config)
        logger.info("SAMPLE CPCB DATA CREATED")
        return

    if args.create_sample_weather:
        create_sample_weather_data(config)
        logger.info("SAMPLE WEATHER DATA CREATED")
        return

    if args.create_sample_ndvi:
        create_sample_ndvi_data(config)
        logger.info("SAMPLE NDVI DATA CREATED")
        return

    if args.create_sample_dem:
        create_sample_dem_data(config)
        logger.info("SAMPLE DEM DATA CREATED")
        return

    if args.create_sample_osm:
        create_sample_osm_data(config)
        logger.info("SAMPLE OSM DATA CREATED")
        return

    if args.create_sample_viirs:
        create_sample_viirs_data(config)
        logger.info("SAMPLE VIIRS DATA CREATED")
        return

    try:
        validate_config(config)

        logger.info("Configuration: PASSED")

        ensure_directories(config)

        logger.info("Directories: PASSED")

        if args.cpcb_only:
            logger.info("=" * 70)
            logger.info("CPCB STAGE ONLY")
            logger.info("=" * 70)
            run_cpcb_pipeline(config)
            logger.info("=" * 70)
            logger.info("CPCB STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.weather_only:
            logger.info("=" * 70)
            logger.info("WEATHER STAGE ONLY")
            logger.info("=" * 70)
            run_weather_pipeline(config)
            logger.info("=" * 70)
            logger.info("WEATHER STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.ndvi_only:
            logger.info("=" * 70)
            logger.info("NDVI STAGE ONLY")
            logger.info("=" * 70)
            run_ndvi_pipeline(config)
            logger.info("=" * 70)
            logger.info("NDVI STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.dem_only:
            logger.info("=" * 70)
            logger.info("DEM STAGE ONLY")
            logger.info("=" * 70)
            run_dem_pipeline(config)
            logger.info("=" * 70)
            logger.info("DEM STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.osm_only:
            logger.info("=" * 70)
            logger.info("OSM STAGE ONLY")
            logger.info("=" * 70)
            run_osm_pipeline(config)
            logger.info("=" * 70)
            logger.info("OSM STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.viirs_only:
            logger.info("=" * 70)
            logger.info("VIIRS STAGE ONLY")
            logger.info("=" * 70)
            run_viirs_pipeline(config)
            logger.info("=" * 70)
            logger.info("VIIRS STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.alignment_only:
            logger.info("=" * 70)
            logger.info("ALIGNMENT STAGE ONLY")
            logger.info("=" * 70)
            run_alignment_pipeline(config)
            logger.info("=" * 70)
            logger.info("ALIGNMENT STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.training_data_only:
            logger.info("=" * 70)
            logger.info("TRAINING DATA STAGE ONLY")
            logger.info("=" * 70)
            run_training_data_pipeline(config)
            logger.info("=" * 70)
            logger.info("TRAINING DATA STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.model_only:
            logger.info("=" * 70)
            logger.info("MODEL TRAINING STAGE ONLY")
            logger.info("=" * 70)
            run_model_pipeline(config)
            logger.info("=" * 70)
            logger.info("MODEL TRAINING STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.inference_only:
            logger.info("=" * 70)
            logger.info("SPATIAL INFERENCE STAGE ONLY")
            logger.info("=" * 70)
            run_inference_pipeline(config, target_date=args.date)
            logger.info("=" * 70)
            logger.info("SPATIAL INFERENCE STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.downscaling_only:
            logger.info("=" * 70)
            logger.info("500 M SPATIAL DOWNSCALING STAGE ONLY")
            logger.info("=" * 70)
            run_downscaling_pipeline(config, target_date=args.date)
            logger.info("=" * 70)
            logger.info("500 M SPATIAL DOWNSCALING STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.aqi_hotspot_only:
            logger.info("=" * 70)
            logger.info("AQI + HOTSPOT + FINAL OUTPUTS STAGE ONLY")
            logger.info("=" * 70)
            run_final_outputs_pipeline(config, target_date=args.date)
            logger.info("=" * 70)
            logger.info("AQI + HOTSPOT + FINAL OUTPUTS STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        if args.global_data_only or args.india_data_only or args.delhi_data_only:
            from src.global_data import run_global_data_pipeline

            scope = "global"
            if args.india_data_only:
                scope = "india"
            elif args.delhi_data_only:
                scope = "delhi"
            logger.info("=" * 70)
            logger.info("GLOBAL DATA ACQUISITION STAGE ONLY (Milestone 16, scope=%s)", scope)
            logger.info("=" * 70)
            run_global_data_pipeline(config, scope=scope)
            logger.info("=" * 70)
            logger.info("GLOBAL DATA ACQUISITION STAGE ONLY: COMPLETED (scope=%s)", scope)
            logger.info("=" * 70)
            return

        if args.global_training_only:
            from src.global_training import run_global_training_pipeline

            logger.info("=" * 70)
            logger.info("GLOBAL TRAINING DATASET STAGE ONLY (Milestone 17, scope=global)")
            logger.info("=" * 70)
            run_global_training_pipeline(config, scope="global")
            logger.info("=" * 70)
            logger.info("GLOBAL TRAINING DATASET STAGE ONLY: COMPLETED")
            logger.info("=" * 70)
            return

        aoi_path = (
            project_root
            / config["study_area"]["boundary_file"]
        )

        if not aoi_path.exists():
            raise FileNotFoundError(
                f"\nAOI missing:\n{aoi_path}\n\n"
                "Place your AOI GeoJSON there first."
            )

        aoi = load_aoi(
            aoi_path,
            config["crs"]["project"]
        )

        logger.info("AOI: PASSED")

        logger.info("=" * 70)
        logger.info("AOD STAGE")
        logger.info("=" * 70)

        try:
            run_aod_pipeline(config)
            logger.info("AOD INGESTION: COMPLETED")
        except Exception as error:
            logger.exception(
                "AOD STAGE FAILED (continuing to CPCB): %s",
                error,
            )

        logger.info("=" * 70)
        logger.info("CPCB STAGE")
        logger.info("=" * 70)

        run_cpcb_pipeline(config)

        logger.info("=" * 70)
        logger.info("ERA5-LAND WEATHER STAGE")
        logger.info("=" * 70)

        run_weather_pipeline(config)

        logger.info("=" * 70)
        logger.info("MODIS NDVI STAGE")
        logger.info("=" * 70)

        run_ndvi_pipeline(config)

        logger.info("=" * 70)
        logger.info("DEM / ELEVATION STAGE")
        logger.info("=" * 70)

        run_dem_pipeline(config)

        logger.info("=" * 70)
        logger.info("OSM ROAD DENSITY STAGE")
        logger.info("=" * 70)

        run_osm_pipeline(config)

        logger.info("=" * 70)
        logger.info("VIIRS NIGHT-TIME LIGHTS STAGE")
        logger.info("=" * 70)

        run_viirs_pipeline(config)

        logger.info("=" * 70)
        logger.info("MASTER DATA ALIGNMENT STAGE")
        logger.info("=" * 70)

        run_alignment_pipeline(config)

        logger.info("=" * 70)
        logger.info("TRAINING DATA / FEATURE ENGINEERING STAGE")
        logger.info("=" * 70)

        run_training_data_pipeline(config)

        logger.info("=" * 70)
        logger.info("MODEL TRAINING / SPATIAL CV STAGE")
        logger.info("=" * 70)

        run_model_pipeline(config)

        logger.info("=" * 70)
        logger.info("1 KM PM2.5 SPATIAL INFERENCE STAGE")
        logger.info("=" * 70)

        run_inference_pipeline(config)

        logger.info("=" * 70)
        logger.info("500 M SPATIAL DOWNSCALING STAGE")
        logger.info("=" * 70)

        run_downscaling_pipeline(config)

        logger.info("=" * 70)
        logger.info("AQI + HOTSPOT + FINAL GEOSPATIAL OUTPUTS STAGE")
        logger.info("=" * 70)

        run_final_outputs_pipeline(config)

        logger.info("=" * 70)
        logger.info("PIPELINE SUMMARY: COMPLETED")
        logger.info("=" * 70)

    except Exception as error:
        logger.exception(
            "PIPELINE FAILED: %s",
            error
        )
        raise


if __name__ == "__main__":
    main()
