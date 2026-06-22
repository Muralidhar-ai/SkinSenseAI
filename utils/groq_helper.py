import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv(override=True)

# Retrieve API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Offline fallback dataset for the disease classes and inconclusive results
OFFLINE_DISEASE_INFO = {
    'Inconclusive': """
        <h3>Analysis Status: Inconclusive</h3>
        <p>The AI analysis is inconclusive because the model's confidence fell below the safety threshold. This indicates that the skin patch image might belong to a condition not included in the model's training database (such as Acne, Eczema, Psoriasis, or Normal skin), or the image quality was insufficient for clear classification.</p>
        
        <h3>General Safe Skincare Guidelines</h3>
        <ul>
            <li><strong>Gentle Cleansing:</strong> Wash the area with a mild, soap-free, non-comedogenic cleanser twice daily. Avoid harsh physical scrubs or rubbing.</li>
            <li><strong>Avoid Picking and Scratching:</strong> Do not squeeze, scratch, or try to pop any lesions or spots, as this can lead to scarring and secondary bacterial infection.</li>
            <li><strong>Sun Protection:</strong> Apply a broad-spectrum sunscreen with SPF 30+ daily to protect the skin barrier and prevent UV-induced inflammation.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>இந்த பகுப்பாய்வின் முடிவு உறுதியற்றதாக உள்ளது. ஏனெனில் உங்கள் தோலில் உள்ள பாதிப்பு இந்த மென்பொருள் பயிற்சி பெற்ற 7 நோய்களுடன் ஒத்துப்போகவில்லை (உதாரணமாக முகப்பரு அல்லது அலர்ஜி போன்றவை). தயவுசெய்து மருத்துவர் அல்லது தோல் மருத்துவரை அணுகி தகுந்த பரிசோதனை செய்து கொள்ளவும்.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>Yes.</strong> Since the automated analysis is inconclusive and cannot identify the condition (which might be acne, eczema, or another out-of-distribution condition), you should consult a certified dermatologist for physical evaluation.</p>
    """,
    'Actinic Keratosis': """
        <h3>Description</h3>
        <p>Actinic Keratosis (also called solar keratosis) is a rough, scaly patch on the skin that develops from years of sun exposure. It is considered a precancerous condition because, if left untreated, it can develop into a type of skin cancer called squamous cell carcinoma.</p>
        
        <h3>Common Causes</h3>
        <ul>
            <li>Chronic and long-term exposure to ultraviolet (UV) radiation from sunlight or tanning beds.</li>
            <li>Cumulative sun damage over several decades, particularly on the face, lips, ears, neck, and forearms.</li>
            <li>Higher susceptibility in fair-skinned individuals or those with weakened immune systems.</li>
        </ul>
        
        <h3>Precautions and Home Care Tips</h3>
        <ul>
            <li>Avoid direct sun exposure during peak hours (10 AM to 4 PM) and seek shade.</li>
            <li>Apply a broad-spectrum sunscreen with an SPF of 30 or higher daily, even on cloudy days.</li>
            <li>Wear protective clothing, such as wide-brimmed hats and UV-blocking sunglasses, when outdoors.</li>
        </ul>
        
        <h3>Allopathy Suggestions</h3>
        <ul>
            <li><strong>La Shield SPF 40 Gel</strong> (Sunscreen) - Apply over exposed skin 20 minutes before sun exposure.</li>
            <li><strong>Sebamed 10% Urea Lotion</strong> (Moisturizer) - Smooths rough patches. Apply twice daily on affected areas.</li>
        </ul>
        <h3>Homeopathy Suggestions</h3>
        <ul>
            <li><strong>Arsenicum Album 30C</strong> - Used for dry, rough, scaly skin patches. Take 2 drops daily.</li>
            <li><strong>Sulphur 30C</strong> - Used for scaly skin lesions. Take under guidance.</li>
        </ul>
        <h3>Siddha Suggestions</h3>
        <ul>
            <li><strong>Parangipattai Choornam</strong> - Traditional blood purifier powder. Take 1 gram with warm milk/water.</li>
            <li><strong>Vetpalai Thailam</strong> - Traditional medicated oil. Apply externally to scaly lesions.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>சூரியக் கதிர்களின் நீண்டகால தாக்கத்தால் தோலில் ஏற்படும் ஒரு வகை கடினமான, செதில் போன்ற தடிப்பு இதுவாகும். இதை கவனிக்காமல் விட்டால், பிற்காலத்தில் இது தோல் புற்றுநோயாக மாற வாய்ப்புள்ளது. சூரிய ஒளியிலிருந்து சருமத்தைப் பாதுகாக்கும் கிரீம்களைப் பயன்படுத்துவது மற்றும் வெயிலில் செல்வதைத் தவிர்ப்பது அவசியமாகும்.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>Yes.</strong> Since Actinic Keratosis is precancerous, it is crucial to consult a dermatologist to evaluate the lesion and discuss removal options such as cryotherapy or topical creams.</p>
    """,
    'Basal Cell Carcinoma': """
        <h3>Description</h3>
        <p>Basal Cell Carcinoma (BCC) is the most common type of skin cancer. It usually develops on sun-exposed areas of the skin, particularly the head and neck, and often appears as a slightly shiny, pearly bump or a sore that does not heal.</p>
        
        <h3>Common Causes</h3>
        <ul>
            <li>Intense, cumulative exposure to ultraviolet (UV) radiation from the sun.</li>
            <li>History of frequent sunburns, particularly during childhood or adolescence.</li>
            <li>Genetic predisposition, fair skin types, and older age.</li>
        </ul>
        
        <h3>Precautions and Home Care Tips</h3>
        <ul>
            <li>Perform regular skin self-exams to detect new growths or changes in existing lesions.</li>
            <li>Protect your skin from UV radiation by using high SPF sunscreens and wearing protective gear.</li>
            <li>Avoid picking, scratching, or trying to self-treat any suspicious skin lesions or non-healing sores.</li>
        </ul>
        
        <h3>Critical Medical Recommendation (No OTC Treatments Available)</h3>
        <ul>
            <li><strong>No over-the-counter (OTC) products or home remedies can treat or cure Basal Cell Carcinoma.</strong></li>
            <li>A dermatologist or oncologist visit is mandatory for a biopsy and diagnostic confirmation.</li>
            <li>Standard medical treatments include surgical excision, Mohs micrographic surgery, cryosurgery, or radiation therapy.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>இது பொதுவாக வெயில் படும் தோலில் தோன்றும் மிகவும் சாதாரணமான ஒரு வகை தோல் புற்றுநோய் ஆகும். இந்த நோய்க்கு வீட்டிலேயே அல்லது கடைகளில் கிடைக்கும் சாதாரண கிரீம்களைக் கொண்டு சுயமாக சிகிச்சை அளிக்க முடியாது. உடனடியாக மருத்துவரை அணுகி அறுவை சிகிச்சை மூலம் இதை அகற்ற வேண்டும்.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>Yes (Mandatory).</strong> A biopsy and prompt surgical excision or other treatments by a qualified dermatologist or oncologist are necessary to prevent local tissue destruction.</p>
    """,
    'Benign Keratosis': """
        <h3>Description</h3>
        <p>Benign Keratosis, commonly referring to Seborrheic Keratoses, is a non-cancerous (benign) skin growth that appears as a waxy or scaly, "stuck-on" plaque. They are very common in older adults and do not pose any cancer risk.</p>
        
        <h3>Common Causes</h3>
        <ul>
            <li>Natural aging process of the skin, typically appearing after the age of 40.</li>
            <li>Genetics and family history of similar skin growths.</li>
            <li>Occasional association with friction in skin folds or mild local irritation.</li>
        </ul>
        
        <h3>Precautions and Home Care Tips</h3>
        <ul>
            <li>Do not pick or scratch the growths, as this can cause bleeding, irritation, or secondary infection.</li>
            <li>Keep the skin well-hydrated to reduce itching or discomfort.</li>
            <li>Consult a doctor if the growth grows rapidly, bleeds, or becomes highly irritated by clothing.</li>
        </ul>
        
        <h3>Allopathy Suggestions</h3>
        <ul>
            <li><strong>Glyco-6 Cream</strong> (Glycolic Acid) - Gently exfoliates and softens rough plaques. Apply thinly at night.</li>
            <li><strong>Venusia Max Cream</strong> (Moisturizer) - Hydrates skin and restores barrier. Apply daily.</li>
        </ul>
        <h3>Homeopathy Suggestions</h3>
        <ul>
            <li><strong>Thuja Occidentalis 30C</strong> - Used for benign waxy skin growths. Take 2 drops with water daily.</li>
            <li><strong>Antimonium Crudum 30C</strong> - Used for hard, scaly growths. Consult a homeopath.</li>
        </ul>
        <h3>Siddha Suggestions</h3>
        <ul>
            <li><strong>Parangipattai Choornam</strong> - Herbal powder for skin scaling. Take 1 gram with warm water.</li>
            <li><strong>Pungan Thailam</strong> - Herbal oil for skin softening. Apply externally to affected area.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>இது தோலில் மெழுகு போன்ற அல்லது வறண்ட ஒட்டினாற்போல் காணப்படும் ஒரு பாதிப்பில்லாத வளர்ச்சி ஆகும். இது புற்றுநோய் அல்லாதது மற்றும் வயது முதிர்ந்தவர்களில் சாதாரணமாகக் காணப்படும். அரிப்பு அல்லது அசௌகரியத்தைத் தவிர்க்க இவற்றை நகங்களால் கிள்ளவோ, சொறியவோ கூடாது.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>No (Conditional).</strong> Usually not required unless the lesion becomes irritated, bleeds, or is cosmetically undesirable, in which case a dermatologist can easily remove it using cryotherapy or electrocautery.</p>
    """,
    'Dermatofibroma': """
        <h3>Description</h3>
        <p>A Dermatofibroma is a common, benign (non-cancerous) skin growth that feels like a small, firm, slightly raised bump. It is typically red, brown, or purple, and often dimples inward when pinched.</p>
        
        <h3>Common Causes</h3>
        <ul>
            <li>Minor skin injuries, such as a bug bite, splinter, or thorn prick.</li>
            <li>Ingrown hairs or minor folliculitis that triggers a localized fibrous reaction.</li>
            <li>Unknown factors, as they frequently develop spontaneously without clear trauma.</li>
        </ul>
        
        <h3>Precautions and Home Care Tips</h3>
        <ul>
            <li>Avoid attempting to squeeze, cut, or puncture the bump, as this will lead to scarring.</li>
            <li>Shave carefully over the area to avoid nicking the bump if it is on the legs.</li>
            <li>Monitor the spot for any changes in size, color, or shape, which should be reported to a doctor.</li>
        </ul>
        
        <h3>Allopathy Suggestions</h3>
        <ul>
            <li><strong>Bio-Oil</strong> (Skincare Oil) - Softens the firm skin lesion and improves appearance. Massage twice daily.</li>
            <li><strong>Nivea Creme</strong> (Moisturizer) - Keeps the bump soft and hydrated. Apply as needed.</li>
        </ul>
        <h3>Homeopathy Suggestions</h3>
        <ul>
            <li><strong>Silicea 30C</strong> - Used for firm fibrous skin nodules. Take 2 drops daily.</li>
            <li><strong>Calcarea Fluorica 6X</strong> - Tissue salt for hardening of tissues. Take 4 tablets twice daily.</li>
        </ul>
        <h3>Siddha Suggestions</h3>
        <ul>
            <li><strong>Sivanar Amirtham</strong> - Traditional formulation for toxic/fibrous skin conditions. Take 100mg with honey.</li>
            <li><strong>Chirattai Thailam</strong> - Medicated oil made from coconut shell. Apply externally to soften the nodule.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>இது தோலில் ஏற்படும் சிறிய, கடினமான, பாதிப்பில்லாத ஒரு வளர்ச்சி ஆகும். பூச்சிக்கடி, முள் குத்துதல் அல்லது சிறிய காயங்களின் எதிர்வினையாக இது உருவாகலாம். இது ஆபத்தானது அல்ல என்றாலும், இதைக் கிள்ளுவதோ அல்லது வெட்ட முயற்சிப்பதோ கூடாது.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>No.</strong> Dermatofibromas are completely harmless and rarely require treatment unless they cause pain, itch intensely, or change rapidly. If diagnostic confirmation is needed, see a doctor.</p>
    """,
    'Melanocytic Nevi': """
        <h3>Description</h3>
        <p>Melanocytic Nevi, commonly known as moles, are benign growths of melanocytes (the cells that produce skin pigment). They can be flat or raised, round or oval, and are usually brown or black in color.</p>
        
        <h3>Common Causes</h3>
        <ul>
            <li>Genetic factors determining the number, size, and location of moles.</li>
            <li>Sun exposure during childhood and adolescence, which increases mole count.</li>
            <li>Hormonal changes, such as during puberty or pregnancy, which can darken or enlarge moles.</li>
        </ul>
        
        <h3>Precautions and Home Care Tips</h3>
        <ul>
            <li>Learn the ABCDE rule of melanoma (Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving).</li>
            <li>Protect your moles from sunburn by using high-quality sunscreens.</li>
            <li>Take photographs of your moles periodically to keep a record and track changes.</li>
        </ul>
        
        <h3>Allopathy Suggestions</h3>
        <ul>
            <li><strong>Re'equil Dry Touch Sunscreen SPF 50</strong> - Protects moles from UV rays and prevents pigment changes.</li>
            <li><strong>Cetaphil Cream</strong> (Moisturizer) - Keeps surrounding skin hydrated. Apply daily.</li>
        </ul>
        <h3>Homeopathy Suggestions</h3>
        <ul>
            <li><strong>Thuja Occidentalis Ointment</strong> - Apply externally on moles/nevi daily.</li>
            <li><strong>Condurango 30C</strong> - Homeopathic remedy for skin spots and nevi. Take 2 drops daily.</li>
        </ul>
        <h3>Siddha Suggestions</h3>
        <ul>
            <li><strong>Kungiliya Vennai</strong> - Traditional cooling ointment. Apply externally over moles to soothe.</li>
            <li><strong>Kasthuri Karuppu</strong> - Herbal mineral compound. Take 100mg under practitioner guidance.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>இவை சாதாரணமாக தோலில் காணப்படும் மச்சங்கள் ஆகும். இவை மெலனோசைட்டுகள் என்ற நிறமி அணுக்கள் ஒரே இடத்தில் குவிவதால் ஏற்படுகின்றன. பெரும்பாலான மச்சங்கள் ஆபத்தற்றவை, ஆனால் அவற்றின் அளவு, வடிவம் அல்லது நிறத்தில் ஏதேனும் மாற்றம் ஏற்பட்டால் மருத்துவரிடம் காண்பிக்க வேண்டும்.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>No (Conditional).</strong> A dermatologist visit is only required if a mole exhibits ABCDE warning signs, bleeds, itches, or changes rapidly, to rule out melanoma.</p>
    """,
    'Melanoma': """
        <h3>Description</h3>
        <p>Melanoma is the most serious type of skin cancer. It develops in the melanocytes, the cells that produce melanin. It often arises in an existing mole or appears as a new, irregular dark spot that changes over time.</p>
        
        <h3>Common Causes</h3>
        <ul>
            <li>Severe, blistering sunburns, particularly at a young age.</li>
            <li>High exposure to ultraviolet (UV) light from sunlight or tanning beds.</li>
            <li>Genetic mutations and family history of melanoma or atypical moles.</li>
        </ul>
        
        <h3>Precautions and Home Care Tips</h3>
        <ul>
            <li>Perform immediate monthly checks using the ABCDE guidelines.</li>
            <li>Avoid any outdoor tanning and protect all exposed skin using hats, clothes, and high SPF sunscreens.</li>
            <li>Consult a dermatologist immediately for any mole that displays itching, bleeding, spreading color, or rapid evolution.</li>
        </ul>
        
        <h3>Critical Medical Recommendation (No OTC Treatments Available)</h3>
        <ul>
            <li><strong>No over-the-counter (OTC) products, creams, or oils can treat or cure Melanoma.</strong></li>
            <li>Consult a dermatologist or surgical oncologist immediately for physical mapping, biopsy, and wide local excision.</li>
            <li>Delaying professional medical care can be extremely dangerous. Early detection and surgical removal are critical.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>மெலனோமா என்பது தோலில் உள்ள நிறமி அணுக்களில் ஏற்படும் மிகவும் கடுமையான தோல் புற்றுநோய் ஆகும். இந்த நோய்க்கு சுயமாக களிம்புகள் கொண்டு சிகிச்சை அளிக்க முடியாது. இது உடலின் மற்ற உறுப்புகளுக்கும் மிக வேகமாகப் பரவக்கூடும் என்பதால், உடனடியாக மருத்துவரை அணுகி அறுவை சிகிச்சை செய்ய வேண்டும்.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>Yes (Immediate/Urgent).</strong> Immediate evaluation, biopsy, and excision by a surgical oncologist or dermatologist are absolutely critical for survival, as melanoma can metastasize quickly.</p>
    """,
    'Vascular Lesion': """
        <h3>Description</h3>
        <p>Vascular Lesions are skin abnormalities related to blood vessels. Common types include cherry angiomas (small red bumps), port-wine stains, hemangiomas, or spider angiomas. They are usually benign and represent clusters of capillaries.</p>
        
        <h3>Common Causes</h3>
        <ul>
            <li>Congenital abnormalities in blood vessel development (present at birth).</li>
            <li>Hormonal changes, pregnancy, or aging processes.</li>
            <li>Minor localized trauma or genetic predisposition.</li>
        </ul>
        
        <h3>Precautions and Home Care Tips</h3>
        <ul>
            <li>Do not squeeze or scratch these lesions, as they contain dilated blood vessels and can bleed easily.</li>
            <li>Keep the skin protected from extreme temperature changes.</li>
            <li>Monitor for changes in size, color, or bleeding patterns.</li>
        </ul>
        
        <h3>Allopathy Suggestions</h3>
        <ul>
            <li><strong>Episoft AM SPF 30</strong> (Sunscreen & Moisturizer) - Gentle daily hydration and UV protection.</li>
            <li><strong>Vicco Turmeric WSO Cream</strong> - Soothes mild vascular inflammation. Apply daily.</li>
        </ul>
        <h3>Homeopathy Suggestions</h3>
        <ul>
            <li><strong>Hamamelis Virginiana 30C</strong> - Relieves dilated blood vessels and capillary clusters. Take 2 drops daily.</li>
            <li><strong>Arnica Montana 30C</strong> - Reduces vascular congestion. Take under guidance.</li>
        </ul>
        <h3>Siddha Suggestions</h3>
        <ul>
            <li><strong>Vetpalai Thailam</strong> - Medicated oil for skin vascularities and inflammation. Apply externally.</li>
            <li><strong>Nannari Chooranam</strong> - Blood purifier and cooling herb. Take 1 gram with warm water.</li>
        </ul>
        
        <h3>Tamil Summary (தமிழ் சுருக்கம்)</h3>
        <p>இவை இரத்தக் குழாய்களின் அசாதாரண வளர்ச்சியால் தோலில் ஏற்படும் சிவப்பு அல்லது ஊதா நிற தழும்புகள் மற்றும் கொப்பளங்கள் ஆகும். இவை பொதுவாக ஆபத்தில்லாதவை, ஆனால் இவற்றிலிருந்து எளிதாக இரத்தம் வரக்கூடும் என்பதால் நகங்களால் கீறுவதோ கிள்ளுவதோ கூடாது.</p>
        
        <h3>Doctor Visit Recommendation</h3>
        <p><strong>No (Conditional).</strong> Usually benign and do not require treatment. Consult a physician for diagnostic confirmation or if you experience bleeding, rapid growth, or wish to seek cosmetic laser removal.</p>
    """
}

def get_disease_info(disease_name):
    """
    Queries the Groq LLaMA 3 model to get detailed structured information about the skin condition.
    If the API call fails or the API key is not configured, it falls back to a highly informative
    offline template for the specified disease.
    """
    
    # Check for Inconclusive flag
    is_inconclusive = False
    if "inconclusive" in disease_name.lower():
        is_inconclusive = True
        matched_key = 'Inconclusive'
    else:
        # Normalize condition name to match offline keys
        matched_key = None
        for k in OFFLINE_DISEASE_INFO.keys():
            if k.lower() in disease_name.lower() or disease_name.lower() in k.lower():
                matched_key = k
                break
                
    fallback_html = OFFLINE_DISEASE_INFO.get(matched_key or 'Inconclusive')
    
    # If the key is not set or is still the default placeholder, skip API call and use local fallback
    if not GROQ_API_KEY or "your_groq_api_key" in GROQ_API_KEY or GROQ_API_KEY == "":
        print(f"[SkinSense AI] Groq API Key not set. Using local offline profile for {disease_name}.")
        return fallback_html

    try:
        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        system_prompt = "You are a professional dermatology assistant. Give accurate, helpful, and safe skin health information."
        
        # Build user prompt according to safety requirements
        if is_inconclusive:
            user_prompt = """The patient's skin image analysis was Inconclusive. The model was not confident or the condition may not be covered in the training dataset.
Please provide:
1. A brief explanation of why the result is inconclusive (1-2 sentences)
2. Precautions and general home care tips (3 bullet points: gentle cleanser, avoid scratching/picking, sunscreen)
3. A Tamil language summary of the condition and advice (3-4 sentences in Tamil)
4. Doctor visit recommendation: Yes or No with reason

Do NOT provide any specific disease names, diagnoses, or over-the-counter medical product recommendations. Provide general supportive advice only.
Format the response in clean HTML with proper headings using <h3> and <ul> tags. Do not wrap in a markdown block, just output the raw HTML."""
        
        elif matched_key in ['Basal Cell Carcinoma', 'Melanoma']:
            user_prompt = f"""The patient's skin image has been analyzed and the predicted condition is: {disease_name}.
This is a malignant/cancerous condition. Do NOT recommend any OTC medical products or home treatments as a cure.
Please provide:
1. A brief description of this condition (2-3 sentences)
2. Common causes (3 bullet points)
3. Precautions and home care tips (3 bullet points)
4. Critical Medical Recommendation: Explain clearly in bullet points that no over-the-counter (OTC) products or home treatments can treat or cure this cancer, and a dermatologist or oncologist visit is mandatory for a biopsy and surgical excision.
5. A Tamil language summary of the condition and advice emphasizing direct medical consult and avoiding self-treatment (3-4 sentences in Tamil)
6. Doctor visit recommendation: Yes or No with reason (Should be Yes, Urgent)

Format the response in clean HTML with proper headings using <h3> and <ul> tags. Do not wrap in a markdown block, just output the raw HTML."""
        
        else:
            user_prompt = f"""The patient's skin image has been analyzed and the predicted condition is: {disease_name}.
Please provide:
1. A brief description of this condition (2-3 sentences)
2. Common causes (3 bullet points)
3. Precautions and home care tips (3 bullet points)
4. Recommended treatment suggestions available in India divided into three separate HTML categories:
   - Allopathy Suggestions (OTC creams, lotions, or tablets with brand name, type, and usage)
   - Homeopathy Suggestions (remedies or tablets with name, potency, and usage)
   - Siddha Suggestions (traditional formulations, choornams, or thailams with name and usage)
5. A Tamil language summary of the condition and advice (3-4 sentences in Tamil)
6. Doctor visit recommendation: Yes or No with reason

Format the response in clean HTML with proper headings using <h3> and <ul> tags. Each treatment category must have its own <h3> heading. Do not wrap in a markdown block, just output the raw HTML."""

        print(f"[SkinSense AI] Calling Groq API with LLaMA-3 for condition: {disease_name}...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=1024
        )
        
        response_content = chat_completion.choices[0].message.content.strip()
        
        # Strip code formatting if LLaMA wraps it in ```html ... ```
        if response_content.startswith("```html"):
            response_content = response_content[7:]
        if response_content.startswith("```"):
            response_content = response_content[3:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
            
        response_content = response_content.strip()
        
        # Verify response is not empty, if it is, raise an exception to hit fallback
        if not response_content:
            raise ValueError("Empty response received from Groq.")
            
        print("[SkinSense AI] Groq API response received successfully.")
        return response_content
        
    except Exception as e:
        print(f"[SkinSense AI] Groq API call failed: {e}. Falling back to offline profiles.")
        return fallback_html
