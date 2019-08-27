// Local support functions

function img_url(algo, mask, by) {
  return "img/train-vs-test" +
    "_alg-" + document.getElementById(algo).value +
    "_mask-" + document.getElementById(mask).value +
    "_by-" + document.getElementById(by).value +
    "_thr-160.png";
}

function perf_img_url(algo, mask, by) {
  return "img/threshold-performance-train-vs-test" +
    "_alg-" + document.getElementById(algo).value +
    "_mask-" + document.getElementById(mask).value +
    "_by-" + document.getElementById(by).value +
    ".png";
}

function gene_list_url(algo, mask, by, which_list) {
  return "gene_lists/genes_from_train-vs-test" +
    "_alg-" + document.getElementById(algo).value +
    "_mask-" + document.getElementById(mask).value +
    "_by-" + document.getElementById(by).value +
    "_" + which_list + ".txt";
}

function img_alt(algo, mask, by) {
  let mask_str = document.getElementById(mask).value;
  if( mask_str !== "none") {
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

function load_file(method, url) {
  return new Promise(function (resolve, reject) {
    let xhr = new XMLHttpRequest();
    xhr.open(method, url);
    xhr.onload = function() {
      if( this.status >= 200 && this.status < 300 ) {
        resolve(xhr.response);
      } else {
        reject({
          status: this.status,
          statusText: xhr.statusText
        });
      }
    };
    xhr.onerror = function() {
      reject({
        status: this.status,
        statusText: xhr.statusText
      });
    };
    xhr.send();
  });
}

async function append_probes_from_file(elementID, algo, mask, by) {
  let text_url = img_url(algo, mask, by);
  text_url = text_url.replace("img/train-vs-test", "rslt/genes-from-train-vs-test");
  text_url = text_url.replace("png", "html");
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
    go_html = gorilla_html("qtpapbc5","al7dno4q", "kzj6m6xg", "7y1quc8e", "zw7chav7");
  } else if((algo === "smrt") && (mask === "64") && (by === "glasser")) {
    go_html = gorilla_html("p9kn0xp8","um6xw6an", "jwo56don", "b88008ft", "ea6uju6w");
  } else if((algo === "smrt") && (mask === "none") && (by === "wellid")) {
    go_html = gorilla_html("n0a3r6fj","dw63o5wr", "i4j4fqtr", "sutcxy7y", "pqm2dlxe");
  } else if((algo === "smrt") && (mask === "64") && (by === "wellid")) {
    go_html = gorilla_html("aq74szoy","ne67jcx5", "blqvcoee", "mlkk99zw", "tmo9fqaa");
  }
  document.getElementById(go_id).innerHTML = gene_html + go_html;
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
    let img_description = img_alt("right_algo","right_mask","right_by");
    document.getElementById("right_image").innerHTML = img_html(
      img_url("right_algo", "right_mask", "right_by"), img_description
    );
    document.getElementById("right_descriptor").innerHTML =
      "<p>" + img_description + "</p>";
    document.getElementById("right_perf_image").innerHTML = img_html(
      perf_img_url("right_algo", "right_mask", "right_by"), img_description
    );
    append_probes_from_file("right_descriptor", "right_algo","right_mask","right_by");
    update_go("right_algo", "right_mask", "right_by", "right_go");
}

function menuDarken( theEvent ) {
	theEvent.target.className = theEvent.target.className.replace("light", "dark");
	theEvent.target.className = theEvent.target.className.replace("black", "dark");
}

function menuLighten( theEvent ) {
	theEvent.target.className = theEvent.target.className.replace("dark", "light");
	theEvent.target.className = theEvent.target.className.replace("black", "light");
}

function initUi( ) {
	console.log("  initializing GE Data Manager UI (in main.js)");

	// For the compare.html view:
    // buildPlot('left_image', 'id_left_set');
    // buildPlot('right_image', 'id_right_set');
    document.getElementById("id_left_set").addEventListener('change', buildPlot('left_image', 'id_left_set'));
    document.getElementById("id_right_set").addEventListener('change', buildPlot('right_image', 'id_right_set'));
}
