// Local support functions

function caption(figure) {
    if( figure === 2) {
        return '<p><span class="heavy">Maximizing Mantel correlations between gene expression similarity ' +
            'and functional connectivity similarity by greedily removing genes.</span> ' +
            '<span class="heavy">A)</span> Mantel correlations for gene expression similarity matrices created ' +
            'from all 15,745 genes are all roughly around zero, as shown in the left-most pane. ' +
            'Repeatedly dropping the gene least supportive ' +
            'of a positive correlation drives the Mantel correlation higher, to a peak, after which dropping ' +
            'any gene results in lower correlation. Black/gray data represent training data, randomly split in half' +
            'by sample. The split-half training data then had sample labels shuffled randomly (green, right-most), ' +
            'weighted to preserve distance (red, center-right), or had its edges shuffled within distance bins ' +
            '(magenta, center-left). Each shuffling paradigm was applied 16 times, each with a different seed. ' +
            'Shuffled data were then subjected to the same Mantel maximization algorithm. Peak Mantel correlations ' +
            'for each set are shown in the right-most pane. ' +
            '<span class="heavy">B)</span> Genes remaining at the peak of each training were more consistent in ' +
            'real data than in shuffled data. Training on randomly shuffled data resulted in randomly selected ' +
            'genes, with low similarity. ' +
            '<span class="heavy">C)</span> Filtering actual data, without shuffling, by the genes discovered in ' +
            'the training phase (with real, shuffled, and/or masked data), resulted in slightly lower correlations, ' +
            'but genes discovered in real data drove higher Mantel correlations than genes discovered in shuffled ' +
            'data. This was true in training data (left-most pane), training data with edges nearer than 16mm ' +
            '(center-left pane), an independent test set (center-right pane), and the test set with proximal edges ' +
            '(< 16mm) removed (right-most pane).' +
            '</p>';
    } else {
        return "<p></p>";
    }
}

function img_url(algo, mask, by) {
    return "img/train-vs-test" +
        "_algo-" + document.getElementById(algo).value +
        "_mask-" + document.getElementById(mask).value +
        "_by-" + document.getElementById(by).value +
        "_thr-160.png";
}

function perf_img_url(algo, mask, by) {
    return "img/threshold-performance-train-vs-test" +
        "_algo-" + document.getElementById(algo).value +
        "_mask-" + document.getElementById(mask).value +
        "_by-" + document.getElementById(by).value +
        ".png";
}

function gene_list_url(algo, mask, by, which_list) {
    return "gene_lists/genes_from_train-vs-test" +
        "_algo-" + document.getElementById(algo).value +
        "_mask-" + document.getElementById(mask).value +
        "_by-" + document.getElementById(by).value +
        "_" + which_list + ".txt";
}

function img_alt(algo, mask, by) {
    let mask_str = document.getElementById(mask).value;
    if (mask_str !== "none") {
        mask_str = "mask &lt; " + mask_str + "mm";
    } else {
        mask_str = "no mask"
    }
    return "Train vs test, " +
        "algo = " + document.getElementById(algo).value +
        ", " + mask_str +
        ", by " + document.getElementById(by).value +
        ", top 160 probes used.";
}

function img_html(url, alt) {
    return "<img " + "style=\"width: 100%;\" " + "src=\"" + url + "\" " + "alt=\"" + alt + "\" />";
}

function gorilla_html(ranked_id, split_id_40, split_id_80, split_id_160, split_id_320) {
    let ranked_url = "http://cbl-gorilla.cs.technion.ac.il/GOrilla/" + ranked_id + "/GOResults.html";
    let split_url_40 = "http://cbl-gorilla.cs.technion.ac.il/GOrilla/" + split_id_40 + "/GOResults.html";
    let split_url_80 = "http://cbl-gorilla.cs.technion.ac.il/GOrilla/" + split_id_80 + "/GOResults.html";
    let split_url_160 = "http://cbl-gorilla.cs.technion.ac.il/GOrilla/" + split_id_160 + "/GOResults.html";
    let split_url_320 = "http://cbl-gorilla.cs.technion.ac.il/GOrilla/" + split_id_320 + "/GOResults.html";
    let html = "<ul>GOrilla results: \n";
    html += "  <li><a href=\"" + ranked_url + "\">all ranked</a></li>\n";
    html += "  <li><a href=\"" + split_url_40 + "\">top 40</a></li>\n";
    html += "  <li><a href=\"" + split_url_80 + "\">top 80</a></li>\n";
    html += "  <li><a href=\"" + split_url_160 + "\">top 160</a></li>\n";
    html += "  <li><a href=\"" + split_url_320 + "\">top 320</a></li>\n";
    html += "</ul>\n";
    return html;
}

function gene_list_html(algo, mask, by) {
    let html = "<ul>Gene lists: \n";
    html += "  <li><a href=\"" +
        gene_list_url(algo, mask, by, "ranked") + "\">all genes ranked</a></li>\n";
    html += "  <li><a href=\"" +
        gene_list_url(algo, mask, by, "foreground-40") + "\">40 foreground</a>, <a href=\"" +
        gene_list_url(algo, mask, by, "background-40") + "\">remaining background</a></li>\n";
    html += "  <li><a href=\"" +
        gene_list_url(algo, mask, by, "foreground-80") + "\">80 foreground</a>, <a href=\"" +
        gene_list_url(algo, mask, by, "background-80") + "\">remaining background</a></li>\n";
    html += "  <li><a href=\"" +
        gene_list_url(algo, mask, by, "foreground-160") + "\">160 foreground</a>, <a href=\"" +
        gene_list_url(algo, mask, by, "background-160") + "\">remaining background</a></li>\n";
    html += "  <li><a href=\"" +
        gene_list_url(algo, mask, by, "foreground-320") + "\">320 foreground</a>, <a href=\"" +
        gene_list_url(algo, mask, by, "background-320") + "\">remaining background</a></li>\n";
    html += "</ul>\n";
    return html;
}

function startResultMining(job_name) {
    document.getElementById("celery_progress_bar").style.display = "block";
    document.getElementById("static_progress_bar").style.display = "none";
    let refreshUrl = "/gedata/REST/refresh/" + job_name;
    let request = new XMLHttpRequest();
    request.onreadystatechange = function () {
        if (request.readyState === 4 && request.status === 200) {
            let responseJsonObj = JSON.parse(request.responseText);
            console.log("Got id: " + responseJsonObj.task_id);
            let progressUrl = "/celery-progress/" + responseJsonObj.task_id + "/";
            CeleryProgressBar.initProgressBar(progressUrl, {
                onSuccess: stopCeleryProcessPolling
            });
        }
    };
    request.open("GET", refreshUrl, true);
    request.send();
    console.log("Beginning celery polling for " + job_name + ".");
}

function stopCeleryProcessPolling() {
    document.getElementById("progress-bar").style.backgroundColor = '#76ce60';
    let peak = document.getElementById("progress-bar-peak-total").innerHTML.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    document.getElementById("progress-bar-message").innerHTML = "Found " + peak + " results.";
    document.getElementById("celery_progress_bar").style.display = "none";
    document.getElementById("static_progress_bar").style.display = "block";
    location.reload();
}

function buildPlot(image_id, select_id) {
    // Calculate the image id string from forms, then use it to load plot images
    let select_element = document.getElementById(select_id);
    select_element.innerText = image_id_from_selections(select_id.toUpperCase()[0]);  // "L" or "R"
    console.log("in buildPlot, select_id = " + select_id + "; containing '" + select_element.innerText + "'.");
    let image_element = document.getElementById(image_id);

    // This function is called a lot just to refresh and update, as well as after changed form fields.
    // If the desired plot is already loaded, we should just ignore the change and quit. No harm done.
    if( image_element.innerHTML.includes(select_element.innerText)) {
        // The desired image is already built and loaded. Nothing to do here.
        console.log("  doing nothing with " + select_element.innerText + "; it's already built and loaded.");
        return;
    } else if (image_element.innerHTML.includes("img src")) {
        // Immediately set the image blank, just to give feedback we registered the click.
        // But don't mess with active spinners (which do not contain "img src" substring)
        console.log("  priming the " + select_id.toUpperCase()[0] + " image spot for " + select_element.innerText + " with empty.");
        loadPlot(image_element, "/static/gedata/empty.png");
    } else if( image_element.innerHTML.includes("fa-spinner")) {
        // Already working on it; ignore further requests.
        // But with async, many requests may be made for the same thing before this is triggered.
        console.log("  " + select_id + " is already building a plot. Wait until it's done!");
        return;
    }

    console.log("Checking for " + select_element.innerText + " image for " + image_id + ".");
    let img_file = "train_test_" + select_element.innerText.toLowerCase() + ".png";
    let img_url = "/static/gedata/plots/" + img_file;

    // The first ajax request determines whether our desired plot already exists or not.
    let png_http = new XMLHttpRequest();
    png_http.onreadystatechange = function () {
        if (png_http.readyState === 4) {
            if (png_http.status === 200) {
                console.log(select_element.innerText + " exists, loading it rather than rebuilding.");
                loadPlot(image_element, img_url);
            }
            if (png_http.status === 404) {
                console.log(select_element.innerText + " does not exist. Building it from scratch.");
                // let refreshUrl = "{% url 'gedata:refresh' job_name=12345 %}".replace(/12345/, img_file);
                let refreshUrl = "/gedata/REST/refresh/" + img_file;

                // The second ajax request initiates image creation and starts the spinner.
                let request = new XMLHttpRequest();
                request.onreadystatechange = function () {
                    if (request.readyState === 4 && request.status === 200) {
                        let responseJsonObj = JSON.parse(this.responseText);
                        if (responseJsonObj.task_id !== "None") {
                            console.log("Got id: " + responseJsonObj.task_id);
                            let progressUrl = "/celery-progress/" + responseJsonObj.task_id + "/";
                            CelerySpinner.initSpinner(progressUrl, {
                                onSuccess: loadPlot,
                                spinnerId: image_id,
                                dataForLater: img_url
                            });
                        }
                    }
                };
                request.open("GET", refreshUrl, true);
                request.send();
                console.log("Beginning to build plot for " + select_element.innerText + " at " + image_id);
            }
        }
    };
    png_http.open('HEAD', img_url, true);
    png_http.send();
}

function assessPerformance(image_id, select_id) {
    // Calculate the image id string from forms, then use it to load plot images
    let select_element = document.getElementById(select_id);
    select_element.innerText = image_id_from_selections(select_id.toLowerCase());  // "performance"
    console.log("in assessPerformance, select_id = " + select_id + "; containing '" + select_element.innerText + "'.");
    let image_element = document.getElementById(image_id);

    // This function is called a lot just to refresh and update, as well as after changed form fields.
    // If the desired plot is already loaded, we should just ignore the change and quit. No harm done.
    if( image_element.innerHTML.includes(select_element.innerText)) {
        // The desired image is already built and loaded. Nothing to do here.
        console.log("  doing nothing with " + select_element.innerText + "; it's already built and loaded.");
        return;
    } else if (image_element.innerHTML.includes("img src")) {
        // Immediately set the image blank, just to give feedback we registered the click.
        // But don't mess with active spinners (which do not contain "img src" substring)
        console.log("  priming the " + select_id.toUpperCase()[0] + " image spot for " + select_element.innerText + " with empty.");
        loadPlot(image_element, "/static/gedata/empty.png");
    } else if( image_element.innerHTML.includes("fa-spinner")) {
        // Already working on it; ignore further requests.
        // But with async, many requests may be made for the same thing before this is triggered.
        console.log("  " + select_id + " is already building a plot. Wait until it's done!");
        return;
    }

    console.log("Checking for " + select_element.innerText + " image for " + image_id + ".");
    let img_file = "performance_" + select_element.innerText.toLowerCase() + ".png";
    let img_url = "/static/gedata/plots/" + img_file;

    // The first ajax request determines whether our desired plot already exists or not.
    let png_http = new XMLHttpRequest();
    png_http.onreadystatechange = function () {
        if (png_http.readyState === 4) {
            if (png_http.status === 200) {
                console.log(select_element.innerText + " exists, loading it rather than rebuilding.");
                loadPlot(image_element, img_url);
            }
            if (png_http.status === 404) {
                console.log(select_element.innerText + " does not exist. Building it from scratch.");
                // let refreshUrl = "{% url 'gedata:refresh' job_name=12345 %}".replace(/12345/, img_file);
                let refreshUrl = "/gedata/REST/refresh/" + img_file;

                // The second ajax request initiates image creation and starts the spinner.
                let request = new XMLHttpRequest();
                request.onreadystatechange = function () {
                    if (request.readyState === 4 && request.status === 200) {
                        let responseJsonObj = JSON.parse(this.responseText);
                        if (responseJsonObj.task_id !== "None") {
                            console.log("Got id: " + responseJsonObj.task_id);
                            let progressUrl = "/celery-progress/" + responseJsonObj.task_id + "/";
                            CelerySpinner.initSpinner(progressUrl, {
                                onSuccess: loadPlot,
                                spinnerId: image_id,
                                dataForLater: img_url
                            });
                        }
                    }
                };
                request.open("GET", refreshUrl, true);
                request.send();
                console.log("Beginning to assess performance for " + select_element.innerText + " at " + image_id);
            }
        }
    };
    png_http.open('HEAD', img_url, true);
    png_http.send();
}

function loadPlot(image_element, image_url) {
    let w = Math.round(document.documentElement.clientWidth * 0.45);
    image_element.style.color = '#ffffff';
    let html_text = "<a href=\"" + image_url + "\">";
    html_text += "<img src=\"" + image_url + "\" width=\"" + w + "\" alt=\"" + image_url + "\">";
    html_text += "</a>";
    image_element.innerHTML = html_text;

    // Also update gene ranking information, if available.
    if( image_url.includes("train_test_") ) {
        if (image_url.endsWith('empty.png')) {
            document.getElementById(image_element.id.replace('image', 'go')).innerHTML = "";
        } else {
            append_probes_from_file(image_element.id.replace('image', 'go'), image_url.replace('png', 'html'));
            document.getElementById(image_element.id.replace('image', 'caption')).innerHTML = caption(2);
        }
    }
}

function load_file(method, url) {
    return new Promise(function (resolve, reject) {
        let xhr = new XMLHttpRequest();
        xhr.open(method, url);
        xhr.onload = function () {
            if (this.status >= 200 && this.status < 300) {
                resolve(xhr.response);
            } else {
                reject({
                    status: this.status,
                    statusText: xhr.statusText
                });
            }
        };
        xhr.onerror = function () {
            reject({
                status: this.status,
                statusText: xhr.statusText
            });
        };
        xhr.send();
    });
}

async function append_probes_from_file(elementID, text_url) {
    console.log(text_url);
    let result = await load_file("GET", text_url);
    document.getElementById(elementID).innerHTML += result;
}

function update_go(algo_id, mask_id, by_id, go_id) {
    // The codes in this function are valid between 6/13/19 and 7/12/19. Then they'll have to be re-run.
    let algo = document.getElementById(algo_id).value;
    let mask = document.getElementById(mask_id).value;
    let by = document.getElementById(by_id).value;
    let gene_html = gene_list_html(algo_id, mask_id, by_id);
    let go_html = "<p>No gene ontology yet.</p>";
    if ((algo === "smrt") && (mask === "none") && (by === "glasser")) {
        go_html = gorilla_html("qtpapbc5", "al7dno4q", "kzj6m6xg", "7y1quc8e", "zw7chav7");
    } else if ((algo === "smrt") && (mask === "64") && (by === "glasser")) {
        go_html = gorilla_html("p9kn0xp8", "um6xw6an", "jwo56don", "b88008ft", "ea6uju6w");
    } else if ((algo === "smrt") && (mask === "none") && (by === "wellid")) {
        go_html = gorilla_html("n0a3r6fj", "dw63o5wr", "i4j4fqtr", "sutcxy7y", "pqm2dlxe");
    } else if ((algo === "smrt") && (mask === "64") && (by === "wellid")) {
        go_html = gorilla_html("aq74szoy", "ne67jcx5", "blqvcoee", "mlkk99zw", "tmo9fqaa");
    }
    document.getElementById(go_id).innerHTML = gene_html + go_html;
}

function image_id_from_selections(side) {
    // For "L" and "R" side for either the "Compare result sets" or "Result set comparisons" page, return the url
    // to the appropriate image, based on form selections.
    let summary_string = "";
    if (side === "L") {
        if( document.title === "Compare result sets" ) {
            let select_element = document.getElementById('id_left_set');
            summary_string = select_element.options[select_element.selectedIndex].value;
        } else if( document.title === "Result set comparisons" ) {
            summary_string += document.getElementById('id_left_comp').value.toLowerCase();
            summary_string += document.getElementById('id_left_parcel').value.toLowerCase()[0];
            summary_string += document.getElementById('id_left_split').value.toLowerCase()[0];
            summary_string += document.getElementById('id_left_train_mask').value;
            summary_string += document.getElementById('id_left_algo').value;
        }
    } else if (side === "R") {
        if( document.title === "Compare result sets" ) {
            let select_element = document.getElementById('id_right_set');
            summary_string = select_element.options[select_element.selectedIndex].value;
        } else if( document.title === "Result set comparisons" ) {
            summary_string += document.getElementById('id_right_comp').value.toLowerCase();
            summary_string += document.getElementById('id_right_parcel').value.toLowerCase()[0];
            summary_string += document.getElementById('id_right_split').value.toLowerCase()[0];
            summary_string += document.getElementById('id_right_train_mask').value;
            summary_string += document.getElementById('id_right_algo').value;
        }
    } else if (side === "center_set_string") {
        if( document.title === "Result set performance" ) {
            summary_string += document.getElementById('id_comp').value.toLowerCase();
            summary_string += document.getElementById('id_parcel').value.toLowerCase()[0];
            summary_string += document.getElementById('id_split').value.toLowerCase()[0];
            summary_string += document.getElementById('id_train_mask').value;
            summary_string += document.getElementById('id_algo').value;
        }
    }
    console.log("    calculated " + side + " summary_string of '" + summary_string + "'");
    return summary_string;
}

function initUi() {
    console.log("  initializing GE Data Manager UI (in main.js)");

    // For the compare.html and comparison.html views:
    //buildPlot('left_image', 'left_set_string');
    //buildPlot('right_image', 'right_set_string');
    if( document.title === "Result set comparisons" ) {
        console.log( "    adding event listeners to form elements.")
        //document.getElementById("id_left_comp").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_parcel").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_split").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_train_mask").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_left_algo").addEventListener('change', buildPlot('left_image', 'left_set_string'));
        //document.getElementById("id_right_comp").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_parcel").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_split").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_train_mask").addEventListener('change', buildPlot('right_image', 'right_set_string'));
        //document.getElementById("id_right_algo").addEventListener('change', buildPlot('right_image', 'right_set_string'));
    } else if ( document.title === "Compare result sets" ) {
        document.getElementById("id_left_set").addEventListener('change', buildPlot('left_image', 'id_left_set'));
        document.getElementById("id_right_set").addEventListener('change', buildPlot('right_image', 'id_right_set'));
    }
    console.log("  completed initing GE Data Manager UI (in main.js)");
}

// Functions called from html events
function update_left() {
    let img_description = img_alt("left_algo", "left_mask", "left_by");
    document.getElementById("left_image").innerHTML = img_html(
        img_url("left_algo", "left_mask", "left_by"), img_description
    );
    document.getElementById("left_descriptor").innerHTML =
        "<p>" + img_description + "</p>";
    document.getElementById("left_perf_image").innerHTML = img_html(
        perf_img_url("left_algo", "left_mask", "left_by"), img_description
    );
    append_probes_from_file("left_descriptor", "left_algo", "left_mask", "left_by");
    update_go("left_algo", "left_mask", "left_by", "left_go");
}

function update_right() {
    let img_description = img_alt("right_algo", "right_mask", "right_by");
    document.getElementById("right_image").innerHTML = img_html(
        img_url("right_algo", "right_mask", "right_by"), img_description
    );
    document.getElementById("right_descriptor").innerHTML =
        "<p>" + img_description + "</p>";
    document.getElementById("right_perf_image").innerHTML = img_html(
        perf_img_url("right_algo", "right_mask", "right_by"), img_description
    );
    append_probes_from_file("right_descriptor", "right_algo", "right_mask", "right_by");
    update_go("right_algo", "right_mask", "right_by", "right_go");
}

