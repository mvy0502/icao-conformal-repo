"""Emit results_v2/mapping_table.csv (TONO -> ICAO/ISO -> BioGaze)."""
import pandas as pd, os
os.makedirs("results_v2", exist_ok=True)
rows = [
 ("bkg",269,"Uniform / plain background","Uniform_background","Yes"),
 ("cap",288,"No head covering","Head_without_covering","Yes"),
 ("ce",244,"Eyes open and visible","Eyes_open","Yes"),
 ("ceg",240,"Eyes open (with spectacles)","Eyes_open","Yes"),
 ("expos",311,"Correct exposure / brightness","Correct_exposure","Yes"),
 ("la_1",308,"Gaze at camera","Gaze_in_camera","Yes"),
 ("la_2",303,"Gaze at camera","Gaze_in_camera","Yes"),
 ("light",149,"Uniform face lighting","Uniform_face_lighting","Yes"),
 ("mkup",197,"No or light makeup","No_or_light_makeup","Yes"),
 ("oof",311,"In focus (sharp)","In_focus","Yes"),
 ("pixel",311,"No pixelation","No_pixelation","Yes"),
 ("poster",311,"No posterization","No_posterization","Yes"),
 ("sat",311,"Correct colour saturation","Correct_saturation","Yes"),
 ("sm",309,"Neutral expression","Neutral_expression","Yes"),
 ("sun",245,"No sunglasses","No_sunglasses","Yes"),
 ("tq",150,"Frontal head pose (yaw)","Frontal_pose","Yes*"),
 ("zoom",311,"Correct face dimension / head-size ratio","— not checked by BioGaze","No"),
]
pd.DataFrame(rows,columns=["tono_folder","n_images","icao_iso_requirement",
    "biogaze_criterion","in_scope_for_biogaze"]).to_csv("results_v2/mapping_table.csv",index=False)
print("wrote results_v2/mapping_table.csv")
